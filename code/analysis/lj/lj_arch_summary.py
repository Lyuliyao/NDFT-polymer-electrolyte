"""LJ benchmark (LJTS r_c 2.5, T 1.5; two identical labels, l_B = 0) of the comparison of
architectures on the ion data (salt_in_polymer_eps2/figs/arch_summary.py), same metrics; densities
0.1 0.2 0.4 0.6 0.7 trained, 0.5 never trained; Gamma(k_1) = 1/S(k_1); no Stillinger-Lovett check.
Original description:
Comparison of architectures on one eps_r: our functional (V2: radial pair
kernels + pointwise map of Gaussian-smoothed densities), the one-body window
network of Sammuller 2023 (c1win) and the equivariant local free energy of
Cheng 2026 (cace), all with the analytic Coulomb part, trained by the same
force matching on the same runs, three seeds each; V1 for reference.

Accuracy: held-out force chi2 and profile error (eps2 without p27, p27
separately), the untrained c = 0.05, Gamma(k_1) at every concentration.
Structure (Table 1 of the paper, on the MD profiles):
  noether   |sum_a int f_a^int dz| / int |f^int| dz, every accepted run (as paper/figures/structural_checks.py)
  jac_asym  ||J - J^T|| / ||J||, J = d mu_theta / d n on the held-out profiles (integrability)
  W_asym    max_k |W_+- - W_-+| / max |W| at uniform density (as structural_checks.py)
  herm      ||K - K^H|| / ||K|| of the complex bulk kernel K(k), k <= 2.1 (integrability and reflection)
  refl      ||mu[Rn] - R mu[n]|| / ||mu - <mu>||, R: z -> -z, every accepted run
  loop      |closed line integral of mu_theta . dn| / sum |legs|, loops nbar -> n1 -> n2 -> nbar over held-out pairs
  SL        |A - 1|, the Stillinger-Lovett intercept (holds for every model with the analytic Coulomb part)
Gamma: 'resp' is the full linear response 2 nbar / |e_N^T M^-1 e_N| with M = nbar^-1 + v_C e e^T + K(k_1)
(complex K), 'sym' uses the Hermitian part of K; both equal the paper's value for a functional.

    SIP_ROOT=<root> python figs/arch_summary.py        (cwd = <root>/code)
"""
import json, os, subprocess, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

R = os.environ["SIP_ROOT"]; L = f"{R}/runs/learn"
sys.path.insert(0, f"{R}/code")
import learn.data as D
from learn.protocol import load_model
from learn.models import mu_theta, force_density, W_of_k, fine_geometry, ESIGN
from learn.geometry import coulomb_vk
from learn.evaluate import stillinger_lovett

T = "c01_c02_c04_c06_c07"
eps2 = False
tag = "lj"
V1 = f"v1_{T}_kbK2"; ours = [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_kbK2_lj_s{s}" for s in (0, 1, 2)]   # 2026-10-04: spectrum loss
CONFIGS = {"V1 (pair closure)": [V1], "ours (V2)": ours}
for lab, arch in (("Sammuller c1, +-3 sigma", "c1win_W30"), ("Sammuller c1, +-6 sigma", "c1win_W60"),
                  ("Cheng CACE, R 1.5 sigma", "cace_a5q3"), ("Cheng CACE, R 3 sigma", "cace_a5q6")):
    CONFIGS[lab] = [f"joint_{arch}_L1_C4_R128x128_H64_{T}_val3_kbK2_{tag}_s{s}" for s in (0, 1, 2)]
path = lambda d: d if d.startswith("/") else os.path.normpath(f"{L}/{d}")
done = lambda d: os.path.exists(f"{path(d)}/predict_c05/metrics.json")

sps, runs = D.load_all()
fam = {(round(r.conc, 4), r.tag): r.family for r in runs}
CS = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7]
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CS}
passed = [r for r in runs if r.status == "PASS" and any(abs(r.conc - c) < 1e-9 for c in CS)]
gm = json.load(open(f"{L}/interpretation/gamma_md.json"))
p27 = lambda r: eps2 and abs(r["conc"] - 0.04) < 1e-9 and r["tag"] == "p27"
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])


def K_complex(ps, cfg, g):
    """Complex bulk kernel K_ab(k) (nk, 2, 2) of mu_theta at the uniform state."""
    N = g.z.shape[0]; n0 = jnp.full((2, N), g.nbar); i0 = N // 2
    f = lambda n: mu_theta(ps, cfg, n, g)
    def one(b):
        t = jnp.zeros((2, N)).at[b, i0].set(1.0)
        _, dmu = jax.jvp(f, (n0,), (t,))
        return jnp.fft.rfft(dmu) / jnp.fft.rfft(t[b])[None, :]
    return np.transpose(np.stack([np.asarray(one(0)), np.asarray(one(1))], -1), (1, 0, 2))


def gamma_k1(K, k, g):
    """(resp, sym) at k_1 from the complex kernel on grid k."""
    i = int(np.argmin(abs(k - 2 * np.pi / g.L)))
    ee = np.outer(ESIGN, ESIGN); e = np.ones(2)
    M = np.eye(2) / g.nbar + coulomb_vk(k[i:i + 1], g.lB)[0] * ee
    Ki = K[i]; Kh = 0.5 * (Ki + Ki.conj().T)
    resp = 2 * g.nbar / abs(e @ np.linalg.solve(M + Ki, e))
    sym = 2 * g.nbar / float(np.real(e @ np.linalg.solve(M + Kh, e)))
    return resp, sym


def Sk_err(K, g, zf):
    """rms relative error of S_pp, S_mm, S_pm (Hermitian part of K) against Sk_grid.npz at k = 2 pi m / L."""
    z = np.load(os.path.join(zf, "Sk_grid.npz"))
    m = np.asarray(z["m"]).astype(int); ks = np.asarray(z["k"])
    ee = np.outer(ESIGN, ESIGN)
    Kh = 0.5 * (K[m] + np.conj(np.swapaxes(K[m], -1, -2)))
    S = np.linalg.inv(np.eye(2)[None] / g.nbar + coulomb_vk(ks, g.lB)[:, None, None] * ee[None] + np.real(Kh))
    mod = np.stack([S[:, 0, 0], S[:, 1, 1], 0.5 * (S[:, 0, 1] + S[:, 1, 0])])
    md = np.stack([z["S_pp"], z["S_mm"], z["S_pm"]])
    scale = np.stack([md[0], md[1], np.sqrt(np.abs(md[0] * md[1]))])
    return 100 * float(np.sqrt(np.mean(((mod - md) / scale) ** 2)))


def structure_checks(ps, cfg, m):
    """Table-1 checks on the MD profiles (see the module docstring)."""
    out = {}
    fd = {c: jax.jit(lambda n, g=g: force_density(ps, cfg, n, g)) for c, g in geoms.items()}
    mt = {c: jax.jit(lambda n, g=g: mu_theta(ps, cfg, n, g)) for c, g in geoms.items()}
    jc = {c: jax.jit(jax.jacfwd(lambda n, g=g: mu_theta(ps, cfg, n, g))) for c, g in geoms.items()}
    noe, refl = [], []
    for r in passed:
        c = round(r.conc, 2); n = jnp.asarray(r.n)
        f = np.asarray(fd[c](n))
        noe.append(abs(f.sum()) / np.abs(f).sum())
        mu = np.asarray(mt[c](n)); muR = np.asarray(mt[c](n[:, ::-1]))[:, ::-1]
        refl.append(np.linalg.norm(muR - mu) / np.linalg.norm(mu - mu.mean(axis=1, keepdims=True)))
    out["noether_max"], out["noether_med"] = float(np.max(noe)), float(np.median(noe))
    out["refl_max"], out["refl_med"] = float(np.max(refl)), float(np.median(refl))
    ho = {(round(r.conc, 4), r.tag) for r in passed} & {(round(x["conc"], 4), x["tag"]) for x in m["heldout_rows"]}
    hor = sorted([r for r in passed if (round(r.conc, 4), r.tag) in ho], key=lambda r: (r.conc, r.tag))
    ja = []
    for r in hor:
        n = jnp.asarray(r.n); N = n.shape[1]
        J = np.asarray(jc[round(r.conc, 2)](n)).reshape(2 * N, 2 * N)
        ja.append(np.linalg.norm(J - J.T) / np.linalg.norm(J))
    out["jac_asym_max"], out["jac_asym_med"] = float(np.max(ja)), float(np.median(ja))
    xg, wg = np.polynomial.legendre.leggauss(16); xg = 0.5 * (xg + 1); wg = 0.5 * wg
    lp = []
    for c in sorted({r.conc for r in hor}):
        rs = [r for r in hor if r.conc == c]
        g = geoms[round(c, 2)]; nb = jnp.full(rs[0].n.shape, g.nbar)
        mu = mt[round(c, 2)]
        leg = lambda a, b: float(sum(w * jnp.sum(mu(a + s * (b - a)) * (b - a)) for s, w in zip(xg, wg)) * g.A * g.dz)
        for i in range(0, len(rs) - 1, 2):
            n1, n2 = jnp.asarray(rs[i].n), jnp.asarray(rs[i + 1].n)
            I = [leg(nb, n1), leg(n1, n2), leg(n2, nb)]
            lp.append(abs(sum(I)) / sum(abs(x) for x in I))
    out["loop_max"], out["loop_med"] = float(np.max(lp)), float(np.median(lp))
    wa, hm, sl = [], [], []
    for c in CS:
        g = geoms[c]
        W = np.asarray(W_of_k(ps, cfg, g)); wa.append(np.abs(W[:, 0, 1] - W[:, 1, 0]).max() / np.abs(W).max())
        K = K_complex(ps, cfg, g); k = np.asarray(g.k); sel = (k > 0) & (k <= 2.1)
        Ks = K[sel]; hm.append(np.linalg.norm(Ks - np.conj(np.swapaxes(Ks, -1, -2))) / np.linalg.norm(Ks))
        sl.append(float("nan"))                 # no charges
    out["W_asym_max"], out["herm_max"], out["SL_max"] = float(max(wa)), float(max(hm)), float(max(sl))
    return out


def model_stats(d, checks=True):
    m = json.load(open(f"{path(d)}/metrics.json")); p = json.load(open(f"{path(d)}/predict_c05/metrics.json"))
    cfg, ps = load_model(path(d))
    ho = [r for r in m["heldout_rows"] if not p27(r)]
    g_ = [r for r in ho if fam.get((round(r["conc"], 4), r["tag"])) == "gauss"]
    s = dict(ho=np.mean([r["chi2"] for r in ho]), L2=np.mean([l2(r) for r in ho]), L2g=np.mean([l2(r) for r in g_]),
             c5=p["transfer"]["chi2"], c5L2=100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"]),
             el=sum(r["el"]["converged"] for r in m["heldout_rows"] + p["transfer_rows"]),
             nel=len(m["heldout_rows"]) + len(p["transfer_rows"]),
             params=sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(ps)), steps=m.get("steps_run", 0),
             fit_min=m.get("fit_seconds", 0) / 60)
    q = [r for r in m["heldout_rows"] if p27(r)]
    if q:
        s["p27"], s["p27L2a"] = q[0]["chi2"], 100 * q[0]["profile"]["anion"]["rel_l2"]
    gam = {}
    for c in CS:
        gf = fine_geometry(geoms[c], 8)
        K = K_complex(ps, cfg, gf)
        gam[c] = gamma_k1(K, np.asarray(gf.k), geoms[c])
        if abs(c - 0.5) < 1e-9:
            s["Sk5"] = Sk_err(K_complex(ps, cfg, geoms[c]), geoms[c], sps[c].zero_field)
    s["gamma"] = gam
    if checks:
        s.update(structure_checks(ps, cfg, m))
    return s


res = {}
for lab, ds in CONFIGS.items():
    ds = [d for d in ds if done(d)]
    if ds:
        res[lab] = [model_stats(d, checks=not lab.startswith("V1")) for d in ds]
        print(f"  {lab}: {len(ds)} models", file=sys.stderr, flush=True)

fmt = lambda v, f=".2f": (f"{np.mean(v):{f}}" if len(v) == 1 else f"{np.mean(v):{f}}±{np.std(v, ddof=1):{f}}")
print(f"== {os.path.basename(R)}" + ("  (held-out without c=0.04 p27)" if eps2 else ""))
cols = [("params", "params", ".0f"), ("ho", "ho chi2", ".2f"), ("L2", "ho L2%", ".2f"), ("L2g", "Gauss L2%", ".2f"),
        ("c5", "rho.5 chi2", ".2f"), ("c5L2", "rho.5 L2%", ".2f"), ("Sk5", "S(k).5 %", ".1f")]
if eps2:
    cols += [("p27", "p27 chi2", ".1f"), ("p27L2a", "p27 anion%", ".1f")]
cols += [("steps", "steps", ".0f"), ("fit_min", "fit min", ".0f")]
print(f"{'':26s}" + "".join(f"{h:>15s}" for _, h, _ in cols) + f"{'EL conv':>10s}")
for lab, ss in res.items():
    print(f"{lab + f' ({len(ss)})':26s}" + "".join(f"{fmt([s[k] for s in ss if k in s], f):>15s}" if any(k in s for s in ss) else f"{'-':>15s}" for k, _, f in cols)
          + f"{sum(s['el'] for s in ss):>6d}/{sum(s['nel'] for s in ss)}")
print("Gamma(k_1): full linear response (deviation in MD sigma); [Hermitian part] where they differ by > 1%")
print(f"{'MD':26s}" + "".join(f"{gm[f'{c:g}']['shell1']['Gamma']:9.2f}±{gm[f'{c:g}']['shell1']['Gamma_err']:.2f}      " for c in CS))
for lab, ss in res.items():
    row = f"{lab:26s}"
    for c in CS:
        md, er = gm[f"{c:g}"]["shell1"]["Gamma"], gm[f"{c:g}"]["shell1"]["Gamma_err"]
        r_ = np.mean([s["gamma"][c][0] for s in ss]); y_ = np.mean([s["gamma"][c][1] for s in ss])
        row += f"{r_:8.2f}({(r_ - md) / er:+5.1f})" + (f"[{y_:5.2f}]" if abs(y_ / r_ - 1) > 0.01 else "       ")
    print(row)
print("Structure on the MD profiles (max over runs / state points; median in parentheses)")
sc = [("noether", "Noether force"), ("jac_asym", "Jacobian asym"), ("W_asym", "W+- vs W-+"), ("herm", "K non-Hermitian"),
      ("refl", "reflection"), ("loop", "loop integral"), ("SL", "Stillinger-Lovett")]
print(f"{'':26s}" + "".join(f"{h:>20s}" for _, h in sc))
for lab, ss in res.items():
    if "noether_max" not in ss[0]:
        continue
    row = f"{lab:26s}"
    for k, _ in sc:
        mx = max(s[f"{k}_max"] for s in ss)
        row += f"{mx:11.1e}" + (f" ({np.median([s[f'{k}_med'] for s in ss]):.0e})" if f"{k}_med" in ss[0] else "         ")
    print(row)
os.makedirs(f"{L}/interpretation", exist_ok=True)
json.dump({lab: [{k: (v if k != "gamma" else {f"{c:g}": list(x) for c, x in v.items()}) for k, v in s.items()} for s in ss]
           for lab, ss in res.items()}, open(f"{L}/interpretation/arch_summary.json", "w"), indent=1, default=float)
