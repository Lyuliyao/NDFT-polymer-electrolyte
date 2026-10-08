"""Learned radial kernels B_m(r)[1 + g(r)] at eps_r = 2: envelope width (widest Gaussian 1-6 sigma) x activation
of Phi and g (silu, gelu, tanh, softplus), three seeds each, same data/protocol as the _lin models.  The choice is
made on the validation residual (mean over seeds), the protocol's own criterion; held-out, c = 0.05, p27 and Gamma
are reported, never used to choose.
    SIP_ROOT=<eps2 root> python figs/knet_scan_summary.py     (cwd = <root>/code)
"""
import json, os, subprocess, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
R = os.environ["SIP_ROOT"]; L = f"{R}/runs/learn"; sys.path.insert(0, f"{R}/code")
import learn.data as D
from learn.protocol import load_model
from learn.evaluate import el_solve
T = "c001_c002_c004_c006_c008"
name = lambda w, a, s: f"joint_v2_L1_C4_R128x128_H64_{T}_val3_" + ("knet" if (w, a) == (2, "silu") else f"kn{w}{a}") + f"_s{s}"
CONF = [(w, a) for w in (1, 2, 3, 4, 6) for a in ("silu", "gelu", "tanh", "softplus")]
REF = [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_lin_s{s}" for s in (0, 1, 2)]       # Gaussians to 2 sigma
done = lambda d: os.path.exists(f"{L}/{d}/predict_c005/metrics.json")
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
gm = json.load(open(f"{L}/interpretation/gamma_md.json")); CG = ["0.01", "0.02", "0.04", "0.05", "0.06", "0.08"]
sps, runs = D.load_all(); sp = sps[0.01]; nbar = sp.n_pairs / sp.L ** 3
lin = [next(x for x in runs if abs(x.conc - 0.01) < 1e-9 and x.tag == t) for t in ("p40", "p41")]

groups = {"Gaussians 2 sigma (reference)": REF}
for w, a in CONF:
    groups[f"learned, {w} sigma, {a}"] = [name(w, a, s) for s in (0, 1, 2)]
groups = {k: [d for d in v if done(d)] for k, v in groups.items()}
allm = [d for v in groups.values() for d in v]
gk = json.loads(subprocess.run([sys.executable, f"{R}/figs/gamma_k1.py"] + [f"{L}/{d}" for d in allm],
                               capture_output=True, text=True, env=dict(os.environ)).stdout)


def stats(d):
    m = json.load(open(f"{L}/{d}/metrics.json")); p = json.load(open(f"{L}/{d}/predict_c005/metrics.json"))
    ho = [r for r in m["heldout_rows"] if r["tag"] != "p27"]
    cfg, ps = load_model(f"{L}/{d}")
    gl = []
    for r in lin:
        t = r.spec["neutral"][0]; g = D.geometry_of(sp, r.n.shape[1]); z = np.asarray(g.z)
        n, _ = el_solve(ps, cfg, g, r.V)
        a = 2 * ((n[0] + n[1]) * np.exp(-1j * t["k"] * z)).mean()
        gl.append(2 * nbar / (-(a * np.exp(-1j * t["phase"])).real / t["A"]))
    return dict(val=m["val_chi2"], ho=np.mean([r["chi2"] for r in ho]), L2=np.mean([l2(r) for r in ho]),
                L2c1=np.mean([l2(r) for r in ho if abs(r["conc"] - 0.01) < 1e-9]),
                p27=next(r["chi2"] for r in m["heldout_rows"] if r["tag"] == "p27"),
                c5=p["transfer"]["chi2"], c5L2=100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"]),
                el=sum(r["el"]["converged"] for r in m["heldout_rows"] + p["transfer_rows"]),
                nel=len(m["heldout_rows"]) + len(p["transfer_rows"]), steps=m.get("steps_run", 0),
                dev=[(gk[d][c] - gm[c]["shell1"]["Gamma"]) / gm[c]["shell1"]["Gamma_err"] for c in CG],
                g01=gk[d]["0.01"], glin=float(np.mean(gl)))


res = {k: [stats(d) for d in v] for k, v in groups.items() if v}
order = sorted(res, key=lambda k: np.mean([s["val"] for s in res[k]]))
f = lambda v, p=2: f"{np.mean(v):.{p}f}±{np.std(v, ddof=1):.{p}f}" if len(v) > 1 else f"{np.mean(v):.{p}f}"
print(f"{'config (sorted by validation chi2)':34s} {'n':>2s} {'validation':>11s} {'held-out':>11s} {'ho L2%':>6s} {'c.01 L2':>7s} "
      f"{'p27':>7s} {'c.05 chi2':>10s} {'c.05 L2':>7s} {'EL':>7s} {'Gamma dev (sigma) c=0.01..0.08':>34s} {'G(.01) seeds':>16s} {'lin k1':>6s} {'steps':>5s}")
for k in order:
    ss = res[k]
    dev = np.mean([s["dev"] for s in ss], axis=0)
    print(f"{k:34s} {len(ss):2d} {f([s['val'] for s in ss], 3):>11s} {f([s['ho'] for s in ss], 3):>11s} {np.mean([s['L2'] for s in ss]):6.2f} "
          f"{np.mean([s['L2c1'] for s in ss]):7.2f} {np.mean([s['p27'] for s in ss]):7.1f} {f([s['c5'] for s in ss], 3):>10s} "
          f"{np.mean([s['c5L2'] for s in ss]):7.2f} {sum(s['el'] for s in ss):3d}/{sum(s['nel'] for s in ss):3d} "
          + " ".join(f"{x:+5.1f}" for x in dev) + "  " + " ".join(f"{s['g01']:.2f}" for s in ss).rjust(16)
          + f" {np.mean([s['glin'] for s in ss]):6.2f} {np.mean([s['steps'] for s in ss]):5.0f}")
print(f"MD Gamma(k_1) c = 0.01: zero field {gm['0.01']['shell1']['Gamma']:.2f} +- {gm['0.01']['shell1']['Gamma_err']:.2f}; "
      f"linear response at k_1 (p40, p41) 0.34 +- 0.05")
json.dump({k: v for k, v in res.items()}, open(f"{L}/interpretation/knet_scan.json", "w"), indent=1, default=float)
