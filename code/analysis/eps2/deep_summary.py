"""eps_r = 2: V2 with re-convolution layers (L = 2, 3) against the quoted L = 1,
three seeds each, and V1.  Held-out force chi2 and profile error (with and
without c = 0.04 p27), the untrained c = 0.05, and Gamma = 2 nbar / S_NN(k_1)
(the paper's definition, figs/gamma_k1.py) against MD.

    SIP_ROOT=<eps2 root> python figs/deep_summary.py      (cwd = <root>/code)
"""
import json, os, subprocess, sys
import numpy as np

R = os.environ["SIP_ROOT"]
L = f"{R}/runs/learn"
T = "c001_c002_c004_c006_c008"
CONFIGS = {
    "V1": [f"v1_{T}_full"],
    "V2 L1 (quoted)": [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_full{s}" for s in ("", "_s1", "_s2")],
    "V2 L2 C4": [f"joint_v2_L2_C4_R128x128_H64_{T}_val3_full_s{s}" for s in (0, 1, 2)],
    "V2 L2 C8": [f"joint_v2_L2_C8_R128x128_H64_{T}_val3_full_s{s}" for s in (0, 1, 2)],
    "V2 L3 C4": [f"joint_v2_L3_C4_R128x128_H64_{T}_val3_full_s{s}" for s in (0, 1, 2)],
}
CS = ["0.01", "0.02", "0.04", "0.05", "0.06", "0.08"]
p27 = lambda r: abs(r["conc"] - 0.04) < 1e-9 and r["tag"] == "p27"
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])

def stats(d):
    m = json.load(open(f"{L}/{d}/metrics.json"))
    p = json.load(open(f"{L}/{d}/predict_c005/metrics.json"))["transfer"]
    ho = m["heldout_rows"]; ok = [r for r in ho if not p27(r)]; q = [r for r in ho if p27(r)][0]
    return dict(train=m["train_chi2"], ho=np.mean([r["chi2"] for r in ho]), ho27=np.mean([r["chi2"] for r in ok]),
                L2=np.mean([l2(r) for r in ok]), c5=p["chi2"], c5L2=100 * 0.5 * (p["cation_rel_l2"] + p["anion_rel_l2"]),
                p27chi=q["chi2"], p27an=100 * q["profile"]["anion"]["rel_l2"], steps=m.get("steps_run", 0),
                ncoef=None)

models = [d for ds in CONFIGS.values() for d in ds if os.path.exists(f"{L}/{d}/predict_c005/metrics.json")]
gk = json.loads(subprocess.run([sys.executable, f"{R}/figs/gamma_k1.py"] + [f"{L}/{d}" for d in models],
                               capture_output=True, text=True, env=dict(os.environ)).stdout)
json.dump(gk, open(f"{L}/gamma_k1_deep.json", "w"), indent=1)
gm = json.load(open(f"{L}/interpretation/gamma_md.json"))

fmt = lambda v: f"{np.mean(v):7.3f}" if len(v) == 1 else f"{np.mean(v):6.3f}±{np.std(v, ddof=1):.3f}"
keys = [("train", "train chi2"), ("ho", "held-out chi2"), ("ho27", "held-out w/o p27"), ("L2", "profile L2 % w/o p27"),
        ("c5", "c=0.05 chi2"), ("c5L2", "c=0.05 L2 %"), ("p27chi", "p27 chi2"), ("p27an", "p27 anion L2 %")]
out = {}
print(f"{'':16s}" + "".join(f"{h:>20s}" for _, h in keys))
for lab, ds in CONFIGS.items():
    ds = [d for d in ds if d in models]
    if not ds: continue
    ss = [stats(d) for d in ds]; out[lab] = ss
    print(f"{lab + f' ({len(ds)})':16s}" + "".join(f"{fmt([s[k] for s in ss]):>20s}" for k, _ in keys))
print("\nGamma(k_1): MD and models (mean over seeds; deviation in MD sigma)")
print(f"{'':16s}" + "".join(f"{c:>18s}" for c in CS))
print(f"{'MD':16s}" + "".join(f"{gm[c]['shell1']['Gamma']:10.2f}±{gm[c]['shell1']['Gamma_err']:.2f}   " for c in CS))
for lab, ds in CONFIGS.items():
    ds = [d for d in ds if d in models]
    if not ds: continue
    row = ""
    for c in CS:
        v = [gk[d][c] for d in ds]; s = gm[c]["shell1"]
        row += f"{np.mean(v):8.2f} ({(np.mean(v) - s['Gamma']) / s['Gamma_err']:+4.1f})  "
    print(f"{lab:16s}" + row)
