"""eps_r = 7.5 confirmation of the basis chosen at eps_r = 2 (learned radial kernels, 1 sigma envelope, softplus)
against the quoted form (Gaussians to 2 sigma, silu), same data, splits, protocol, three seeds each.
Criteria fixed before the runs finished: the new basis is confirmed if (i) its held-out residual, profile error
and c = 0.05 residual are not worse than the quoted form's by more than the seed spread, (ii) its Gamma(k_1)
deviations from MD are not larger, and (iii) it is not worse on the strongest well (validation run c = 0.08 p01).
    SIP_ROOT=<eps75 root> python figs/kern75_compare.py      (cwd = <root>/code)
"""
import json, os, subprocess, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
R = os.environ["SIP_ROOT"]; L = f"{R}/runs/learn"; sys.path.insert(0, f"{R}/code")
import learn.data as D
from learn.protocol import load_model
from learn.evaluate import heldout_report
T = "c001_c002_c004_c006_c008"
SETS = {"quoted (Gaussians 2 sigma, silu)": [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_long_s{s}" for s in (0, 1, 2)],
        "learned 1 sigma, softplus": [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus_long_s{s}" for s in (0, 1, 2)]}
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
f = lambda v, p=3: f"{np.mean(v):.{p}f}±{np.std(v, ddof=1):.{p}f}" if len(v) > 1 else f"{np.mean(v):.{p}f}"
CS = [0.01, 0.02, 0.04, 0.06, 0.08]; CG = ["0.01", "0.02", "0.04", "0.05", "0.06", "0.08"]
gpath = f"{L}/interpretation/gamma_md.json"
gm = json.load(open(gpath if os.path.exists(gpath) else "/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn/interpretation/gamma_md.json"))
sps, runs = D.load_all()
hard = [r for r in runs if abs(r.conc - 0.08) < 1e-9 and r.tag == "p01"]
ok = {k: [d for d in v if os.path.exists(f"{L}/{d}/predict_c005/metrics.json")] for k, v in SETS.items()}
gk = json.loads(subprocess.run([sys.executable, "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/figs/gamma_k1.py"]
                               + [f"{L}/{d}" for v in ok.values() for d in v], capture_output=True, text=True, env=dict(os.environ)).stdout)
print(f"{'':34s} {'validation':>13s} {'held-out':>13s}  by c (0.01..0.08)                {'profile %':>9s} {'c=0.05 chi2':>13s} {'c=0.05 %':>8s} {'c0.08 p01':>9s}")
for lab, ds in ok.items():
    if not ds:
        print(f"{lab}: not done"); continue
    M = [json.load(open(f"{L}/{d}/metrics.json")) for d in ds]; P = [json.load(open(f"{L}/{d}/predict_c005/metrics.json")) for d in ds]
    byc = [np.mean([np.mean([r["chi2"] for r in m["heldout_rows"] if abs(r["conc"] - c) < 1e-9]) for m in M]) for c in CS]
    hp = []
    for d in ds:
        cfg, ps = load_model(f"{L}/{d}")
        hp.append(heldout_report(ps, cfg, D.make_batches(hard, sps), el=True, log=None)[0])
    print(f"{lab:34s} {f([m['val_chi2'] for m in M]):>13s} {f([m['heldout_chi2'] for m in M]):>13s}  " + " ".join(f"{x:.3f}" for x in byc)
          + f"  {np.mean([m['heldout_rel_l2'] * 100 for m in M]):9.2f} {f([p['transfer']['chi2'] for p in P]):>13s} "
          f"{np.mean([100 * 0.5 * (p['transfer']['cation_rel_l2'] + p['transfer']['anion_rel_l2']) for p in P]):8.2f} "
          f"{np.mean([h['chi2'] for h in hp]):9.1f} (anion {np.mean([100 * h['profile']['anion']['rel_l2'] for h in hp]):.1f}%)")
print("Gamma(k_1), deviation in MD sigma")
print(f"{'MD':34s}" + "".join(f"{gm[c]['shell1']['Gamma']:7.2f}±{gm[c]['shell1']['Gamma_err']:.2f}  " for c in CG))
for lab, ds in ok.items():
    if ds:
        g = {c: [gk[d][c] for d in ds] for c in CG}
        print(f"{lab:34s}" + "".join(f"{np.mean(g[c]):7.2f}({(np.mean(g[c]) - gm[c]['shell1']['Gamma']) / gm[c]['shell1']['Gamma_err']:+5.1f})" for c in CG)
              + "   c=0.01 seeds " + " ".join(f"{x:.2f}" for x in g["0.01"]))
