"""Two-dimensional test: the functionals trained on planar profiles predict n_a(x, y) under potentials
V_a(x, y) (learn.field2d, nothing retrained), compared with MD (tools/analyze_profiles2d.py).

Systems: c = 0.04 at eps_r = 7.5 and 2 (state dirs <scratch>/field2d/eps{75,2}_c0.04, potentials p50-p53).
Models (2026-10-04: spectrum loss p = 2, _kbK2): the paper's neural functional (learned kernels, 3 seeds), its Gaussian form
(3 seeds), the pair closure with the same learned kernels (2026-10-08; before: Gaussian basis, closed form), Poisson-Boltzmann.  Predictions are cached in <research>/salt_in_polymer_field2d/.
Metrics on the Fourier amplitudes a(m_x, m_y) of each species, |m| <= 15, MD errors from the block series
with their integrated autocorrelation time; modes with |a_MD| > 3 err are used:
  amp     sum |a_pred - a_MD| / sum |a_MD|             (all significant modes; driven; undriven = generated
                                                        by the nonlinear coupling of the driven ones)
  L2      band-limited relative L2 error of the map, sqrt(sum |da|^2) / ||n_MD||, with the MD noise floor
Without MD data only the predictions are made (convergence, density range).
    python figs/field2d_compare.py [--predict-only]
"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
B = "/mnt/research/MultiscaleML_group/Liyao"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
sys.path.insert(0, f"{B}/salt_in_polymer_eps2/code"); sys.path.insert(0, f"{S}/tools")
os.environ.setdefault("SIP_ROOT", f"{B}/salt_in_polymer_eps2")
from learn.protocol import load_model
from learn.models import ModelConfig, init_params
from learn import field2d as F2
from analyze_profiles2d import pot
from field_acceptance import tau_int

T = "c001_c002_c004_c006_c008"; OUT = f"{B}/salt_in_polymer_field2d"
SYS = {
    "eps75": dict(L=24.502285, lB=7.957187947537492, state=f"{S}/field2d/eps75_c0.04",
                  models={**{f"NF s{s}": f"{B}/salt_in_polymer_eps75/runs/learn/joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus_kbK2_long_s{s}" for s in (0, 1, 2)},
                          **{f"Gauss s{s}": f"{B}/salt_in_polymer_eps75/runs/learn/joint_v2_L1_C4_R128x128_H64_{T}_val3_kbK2_long_s{s}" for s in (0, 1, 2)},
                          "pair closure": f"{B}/salt_in_polymer_eps75/runs/learn/joint_v1_L1_C4_R128x128_H64_{T}_val3_kn1softplus_kbK2_long_s2"}),
    "eps2": dict(L=24.416564, lB=float(np.load(f"{S}/eps2/runs/prod_T1.0_c0.04/zero_field/Sk.npz")["lB"]), state=f"{S}/field2d/eps2_c0.04",
                 models={**{f"NF s{s}": f"{B}/salt_in_polymer_eps2/runs/learn/joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus_kbK2_s{s}" for s in (0, 1, 2)},
                         **{f"Gauss s{s}": f"{B}/salt_in_polymer_eps2/runs/learn/joint_v2_L1_C4_R128x128_H64_{T}_val3_kbK2_full_s{s}" for s in (0, 1, 2)},
                         "pair closure": f"{B}/salt_in_polymer_eps2/runs/learn/joint_v1_L1_C4_R128x128_H64_{T}_val3_kn1softplus_kbK2_s0"}),
}
TAGS = ("p50", "p51", "p52", "p53", "p54", "p55")
GROUPS = (("neural functional", ("NF s0", "NF s1", "NF s2")), ("Gaussian form", ("Gauss s0", "Gauss s1", "Gauss s2")),
          ("pair closure", ("pair closure",)), ("Poisson-Boltzmann", ("PB",)))


def err_ac(x):
    f = lambda v: v.var(ddof=1) * 2 * tau_int(v) / len(v)
    return np.sqrt(f(x.real) + f(x.imag))


def predict(sysname, tag):
    sy = SYS[sysname]; os.makedirs(f"{OUT}/{sysname}", exist_ok=True)
    spec = json.load(open(f"{sy['state']}/potentials/{tag}.json"))
    N = int(round(sy["L"] / 0.1)); g = F2.make_geometry2d(sy["L"], N, 480, sy["lB"])
    mdf = f"{sy['state']}/field_{tag}/profiles2d.npz"
    md = np.load(mdf, allow_pickle=True) if os.path.exists(mdf) else None
    lo = md["lo"] if md is not None else np.zeros(3)
    cache = f"{OUT}/{sysname}/{tag}_pred.npz"
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        if np.allclose(c["lo"], lo) and all(f"n_{lab}" in c.files for lab in list(sy["models"]) + ["PB"]):
            return spec, g, md, {k[2:]: c[k] for k in c.files if k.startswith("n_")}, json.loads(str(c["info"]))
    x = (np.arange(N) + 0.5) * sy["L"] / N; X, Y = np.meshgrid(x, x, indexing="ij")
    V = np.stack([pot(spec, 1.0, X + lo[0], Y + lo[1]), pot(spec, -1.0, X + lo[0], Y + lo[1])])
    pred, info = {}, {}
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        if np.allclose(c["lo"], lo):
            pred = {k[2:]: c[k] for k in c.files if k.startswith("n_")}; info = json.loads(str(c["info"]))
    for lab, d in {**sy["models"], "PB": None}.items():
        if lab in pred: continue
        if d is not None and not os.path.exists(os.path.join(d, "params.pkl")):
            print(f"  {sysname} {tag}: {lab} not trained yet, skipped"); continue
        cfg, ps = (ModelConfig("pb"), init_params(ModelConfig("pb"))) if d is None else load_model(d)
        pred[lab], info[lab] = F2.el_solve(ps, cfg, g, V)
    np.savez_compressed(cache, lo=lo, info=json.dumps(info), **{f"n_{k}": v for k, v in pred.items()})
    return spec, g, md, pred, info


def modes_of(n, mx, my, L):
    """a(m_x, m_y) = (1/N^2) sum n(x_c, y_c) exp(-i k.r_c) at the bin centres, for one species."""
    N = n.shape[0]; xc = (np.arange(N) + 0.5) * L / N; k1 = 2 * np.pi / L
    Ex = np.exp(-1j * k1 * xc[:, None] * mx[None, :]); Ey = np.exp(-1j * k1 * xc[:, None] * my[None, :])
    return Ex.T @ n @ Ey / N ** 2


out = {}
for sysname in SYS:
    for tag in TAGS:
        spec, g, md, pred, info = predict(sysname, tag)
        nb = g.nbar
        line = f"{sysname} {tag} {spec['family']:10s} {spec['kind']:8s}"
        if md is None:
            print(line + "  (no MD yet)  " + "  ".join(
                f"{lab}: n/nbar [{pred[lab].min() / nb:.2f}, {pred[lab].max() / nb:.1f}] {'ok' if info[lab]['converged'] else 'EL FAIL'} {info[lab]['iters']}it"
                for lab in ("NF s0", "Gauss s0", "pair closure", "PB")))
            continue
        mx, my, drv = md["mode_mx"], md["mode_my"], md["driven"]
        half = (my[None, :] > 0) | ((my[None, :] == 0) & (mx[:, None] > 0))
        res = {}
        acc = json.load(open(f"{SYS[sysname]['state']}/field_{tag}/acceptance.json"))
        print(line + f"  MD {'PASS' if acc['pass'] else 'REVIEW'}, contrast {acc['cation']['contrast']:.1f} / {acc['anion']['contrast']:.1f}")
        for glab, labs in GROUPS:
            rows = []
            labs = [l for l in labs if l in pred]
            if not labs: continue
            for lab in labs:
                r = {}
                for a, sp in enumerate(("cation", "anion")):
                    ser = md[f"{sp}_mode_n"]; am = ser.mean(0)
                    er = np.array([[err_ac(ser[:, i, j]) for j in range(len(my))] for i in range(len(mx))])
                    sig = half & (np.abs(am) > 3 * er)
                    da = modes_of(pred[lab][a], mx, my, g.L) - am
                    den = np.sqrt(nb ** 2 + 2 * np.sum(np.abs(am[half]) ** 2))
                    for nm, sel in (("all", sig), ("driven", sig & drv), ("undriven", sig & ~drv)):
                        r.setdefault(f"amp_{nm}", []).append(100 * np.abs(da[sel]).sum() / max(np.abs(am[sel]).sum(), 1e-300))
                        r.setdefault(f"n_{nm}", []).append(int(sel.sum()))
                    r.setdefault("L2", []).append(100 * np.sqrt(2 * np.sum(np.abs(da[half]) ** 2)) / den)
                    r.setdefault("floor", []).append(100 * np.sqrt(2 * np.sum(er[half] ** 2)) / den)
                    r.setdefault("pull", []).append(float(np.sqrt(np.mean((np.abs(da[sig]) / er[sig]) ** 2))))
                rows.append({k: float(np.mean(v)) for k, v in r.items()} | {"converged": info[lab]["converged"]})
            m = lambda k: np.mean([x[k] for x in rows])
            rng = lambda k: (f" [{min(x[k] for x in rows):.1f}-{max(x[k] for x in rows):.1f}]" if len(rows) > 1 else "")
            print(f"    {glab:18s} amplitude error {m('amp_all'):5.1f}%{rng('amp_all')}  driven {m('amp_driven'):5.1f}%  undriven {m('amp_undriven'):5.1f}% "
                  f"({m('n_driven'):.0f} + {m('n_undriven'):.0f} modes)   map L2 {m('L2'):5.2f}% (MD noise {m('floor'):.2f}%)   rms pull {m('pull'):5.1f}"
                  + ("" if all(x["converged"] for x in rows) else "   EL not converged"))
            res[glab] = rows
        out[f"{sysname}/{tag}"] = res
json.dump(out, open(f"{OUT}/field2d_summary.json", "w"), indent=1)
