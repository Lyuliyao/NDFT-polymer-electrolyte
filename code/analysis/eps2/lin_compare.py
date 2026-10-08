"""eps_r = 2: the quoted models (_full) against the same configuration trained with the two linear-regime runs
at c = 0.01 (p40, p41) added (_lin): held-out residual and profile error by concentration (p27 apart), the
untrained c = 0.05, Gamma(k_1) against MD, and the linear response on p40-p43 (p42/p43 never trained).
    SIP_ROOT=<eps2 root> python figs/lin_compare.py      (cwd = <root>/code)
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
SETS = {"V2 quoted": [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_full{s}" for s in ("", "_s1", "_s2")],
        "V2 + p40/p41": [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_lin_s{s}" for s in (0, 1, 2)],
        "V1 quoted": [f"v1_{T}_full"], "V1 + p40/p41": [f"v1_{T}_lin"]}
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
fmt = lambda v: f"{np.mean(v):6.3f}" + (f"±{np.std(v, ddof=1):.3f}" if len(v) > 1 else "      ")
CS = [0.01, 0.02, 0.04, 0.06, 0.08]
gm = json.load(open(f"{L}/interpretation/gamma_md.json"))
ok = {k: [d for d in v if os.path.exists(f"{L}/{d}/predict_c005/metrics.json")] for k, v in SETS.items()}
print("held-out chi2 by concentration (p27 apart) | profile L2 % | c=0.05 chi2 / L2 %")
for lab, ds in ok.items():
    if not ds:
        print(f"  {lab}: not done"); continue
    M = [json.load(open(f"{L}/{d}/metrics.json")) for d in ds]; P = [json.load(open(f"{L}/{d}/predict_c005/metrics.json")) for d in ds]
    by = lambda m, c, f: np.mean([f(r) for r in m["heldout_rows"] if abs(r["conc"] - c) < 1e-9 and r["tag"] != "p27"])
    chi = "  ".join(f"{fmt([by(m, c, lambda r: r['chi2']) for m in M])}" for c in CS)
    prof = "  ".join(f"{np.mean([by(m, c, l2) for m in M]):5.2f}" for c in CS)
    p27 = [r["chi2"] for m in M for r in m["heldout_rows"] if r["tag"] == "p27"]
    c5 = f"{fmt([p['transfer']['chi2'] for p in P])} / {np.mean([100 * 0.5 * (p['transfer']['cation_rel_l2'] + p['transfer']['anion_rel_l2']) for p in P]):.2f}"
    print(f"  {lab:14s} chi2 {chi}   L2 {prof}   p27 {np.mean(p27):5.1f}   c=0.05 {c5}")
out = json.loads(subprocess.run([sys.executable, f"{R}/figs/gamma_k1.py"] + [f"{L}/{d}" for v in ok.values() for d in v],
                                capture_output=True, text=True, env=dict(os.environ)).stdout)
print("Gamma(k_1) = 2 nbar / S_NN(k_1), deviation in MD sigma (zero field)")
CG = ["0.01", "0.02", "0.04", "0.05", "0.06", "0.08"]
print(f"  {'MD':14s}" + "".join(f"{gm[c]['shell1']['Gamma']:7.2f}±{gm[c]['shell1']['Gamma_err']:.2f}    " for c in CG))
for lab, ds in ok.items():
    if ds:
        g = {c: [out[d][c] for d in ds] for c in CG}
        print(f"  {lab:14s}" + "".join(f"{np.mean(g[c]):7.2f}({(np.mean(g[c]) - gm[c]['shell1']['Gamma']) / gm[c]['shell1']['Gamma_err']:+5.1f})    " for c in CG)
              + ("   c=0.01 seeds " + " ".join(f"{x:.2f}" for x in g["0.01"]) if len(ds) > 1 else ""))
# linear response on p40-p43
sps, runs = D.load_all(); sp = sps[0.01]; nbar = sp.n_pairs / sp.L ** 3
lin = json.load(open(f"{L}/interpretation/linear_gamma_c001.json"))
print("linear response Gamma_lin (MD from figs/linear_gamma.py; models: Euler-Lagrange, seed mean [range])")
for tag in ("p40", "p41", "p42", "p43"):
    r = next(x for x in runs if abs(x.conc - 0.01) < 1e-9 and x.tag == tag)
    t = r.spec["neutral"][0]; g = D.geometry_of(sp, r.n.shape[1]); z = np.asarray(g.z)
    row = f"  {tag} k_{t['m']} A {t['A']:.2f} [{r.status}]  MD {lin[tag]['Gamma_lin']:.2f}±{lin[tag]['Gamma_lin_err']:.2f}"
    for lab in ("V2 quoted", "V2 + p40/p41"):
        vals = []
        for d in ok[lab]:
            cfg, ps = load_model(f"{L}/{d}")
            n, _ = el_solve(ps, cfg, g, r.V)
            a = 2 * ((n[0] + n[1]) * np.exp(-1j * t["k"] * z)).mean()
            vals.append(2 * nbar / (-(a * np.exp(-1j * t["phase"])).real / t["A"]))
        if vals:
            row += f"   {lab}: {np.mean(vals):.2f} [{min(vals):.2f}-{max(vals):.2f}]"
    print(row)
