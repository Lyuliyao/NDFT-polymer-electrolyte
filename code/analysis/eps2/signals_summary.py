"""Comparison of training signals on one eps_r: force matching (ours), local
chemical-potential balance (lmu, Cheng 2026) and pair-correlation matching
(pcm, Dijkman 2025), same architecture (V2 L1 R128x128), three seeds each, V1
for reference.  Held-out force chi2 and profile error (split into Gaussian
wells and the rest), the untrained c = 0.05 (profiles and S_ab(k)), and
Gamma(k_1) at every concentration.

    SIP_ROOT=<eps root> python figs/signals_summary.py <force|tag> ...   (cwd = <root>/code)
"""
import json, os, subprocess, sys
import numpy as np
import jax.numpy as jnp

R = os.environ["SIP_ROOT"]; L = f"{R}/runs/learn"
sys.path.insert(0, f"{R}/code")
import learn.data as D
from learn.protocol import load_model
from learn.signals import PCMTarget, pcm_model_S

T = "c001_c002_c004_c006_c008"
eps2 = "eps2" in os.path.basename(R)
if eps2:
    V1 = f"v1_{T}_full"; force = [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_full{s}" for s in ("", "_s1", "_s2")]; tag = "full"
else:
    V1 = f"/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn/v1_{T}_long"
    force = [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_long_s{s}" for s in (0, 1, 2)]; tag = "long"
CONFIGS = {"V1 (pair kernel)": [V1], "force matching (ours)": force}
for loss, lab in (("lmu", "local-mu balance"), ("pcm", "pair-corr. matching")):
    CONFIGS[lab] = [f"joint_v2_L1_C4_R128x128_H64_{T}_val3_{loss}_{tag}_s{s}" for s in (0, 1, 2)]
path = lambda d: d if d.startswith("/") else os.path.normpath(f"{L}/{d}")

p27 = lambda r: eps2 and abs(r["conc"] - 0.04) < 1e-9 and r["tag"] == "p27"
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
fam = {}
sps, runs = D.load_all()
for r in runs: fam[(round(r.conc, 4), r.tag)] = r.family
# S_ab(k) at the untrained c = 0.05
sp5 = sps[0.05]; N5 = next(r.n.shape[1] for r in runs if abs(r.conc - 0.05) < 1e-9)
t5 = PCMTarget(sp5, D.geometry_of(sp5, N5), rel_floor=0.0)

def stats(d):
    m = json.load(open(f"{path(d)}/metrics.json")); p = json.load(open(f"{path(d)}/predict_c005/metrics.json"))
    ho = [r for r in m["heldout_rows"] if not p27(r)]
    g = [r for r in ho if fam.get((round(r["conc"], 4), r["tag"])) == "gauss"]; o = [r for r in ho if r not in g]
    cfg, prm = load_model(path(d))
    S = np.asarray(pcm_model_S(prm, cfg, t5)); z = np.asarray(t5.S)
    scale = np.stack([z[0], z[1], np.sqrt(np.abs(z[0] * z[1]))])
    sk = 100 * np.sqrt(np.mean(((S - z) / scale) ** 2))
    return dict(ho=np.mean([r["chi2"] for r in ho]), L2=np.mean([l2(r) for r in ho]),
                L2g=np.mean([l2(r) for r in g]), L2o=np.mean([l2(r) for r in o]),
                c5=p["transfer"]["chi2"], c5L2=100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"]),
                sk5=sk, steps=m.get("steps_run", 0))

models = [d for ds in CONFIGS.values() for d in ds if os.path.exists(f"{path(d)}/predict_c005/metrics.json")]
gk = json.loads(subprocess.run([sys.executable, f"{R}/../salt_in_polymer_eps2/figs/gamma_k1.py"] + [path(d) for d in models],
                               capture_output=True, text=True, env=dict(os.environ)).stdout)
gm = json.load(open(f"{L}/interpretation/gamma_md.json")) if os.path.exists(f"{L}/interpretation/gamma_md.json") else \
     json.load(open("/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn/interpretation/gamma_md.json"))
fmt = lambda v: f"{np.mean(v):6.2f}" if len(v) == 1 else f"{np.mean(v):5.2f}±{np.std(v, ddof=1):.2f}"
keys = [("ho", "held-out chi2"), ("L2", "held-out L2%"), ("L2g", "  Gauss wells"), ("L2o", "  other runs"),
        ("c5", "c=0.05 chi2"), ("c5L2", "c=0.05 L2%"), ("sk5", "S_ab(k) c=0.05 %"), ("steps", "steps")]
print(f"== {os.path.basename(R)}" + ("  (held-out without c=0.04 p27)" if eps2 else ""))
print(f"{'':24s}" + "".join(f"{h:>18s}" for _, h in keys))
for lab, ds in CONFIGS.items():
    ds = [d for d in ds if d in models]
    if not ds: continue
    ss = [stats(d) for d in ds]
    print(f"{lab + f' ({len(ds)})':24s}" + "".join(f"{fmt([s[k] for s in ss]):>18s}" for k, _ in keys))
CS = ["0.01", "0.02", "0.04", "0.05", "0.06", "0.08"]
print("Gamma(k_1) (deviation in MD sigma)")
print(f"{'MD':24s}" + "".join(f"{gm[c]['shell1']['Gamma']:9.2f}±{gm[c]['shell1']['Gamma_err']:.2f}" for c in CS))
for lab, ds in CONFIGS.items():
    ds = [d for d in ds if d in models]
    if not ds: continue
    print(f"{lab:24s}" + "".join(f"{np.mean([gk[os.path.basename(path(d))][c] for d in ds]):8.2f}({(np.mean([gk[os.path.basename(path(d))][c] for d in ds]) - gm[c]['shell1']['Gamma']) / gm[c]['shell1']['Gamma_err']:+5.1f})" for c in CS))
