"""LJ benchmark, training-set size: ours, Sammuller (+-3 sigma) and Cheng (R 1.5 sigma), the windows
chosen by validation chi2, retrained on nested stratified fractions of the training runs (3 seeds)."""
import json, os
import numpy as np
L = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj/runs/learn"; T = "c01_c02_c04_c06_c07"
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
print(f"{'':30s}{'train runs':>12s}{'ho chi2':>14s}{'ho median':>12s}{'ho L2%':>12s}{'rho.5 chi2':>14s}{'rho.5 L2%':>12s}")
for frac, suf in (("0.25", "_f025"), ("0.5", "_f050"), ("1", "")):
    for lab, a in (("ours, learned kernels", "v2"), ("ours, Gaussian", "v2"), ("Sammuller +-3", "c1win_W30"), ("Cheng R1.5", "cace_a5q3")):
        v = []
        for s in (0, 1, 2):
            d = f"{L}/joint_{a}_L1_C4_R128x128_H64_{T}_val3{'_kn1softplus' if lab.endswith('kernels') else ''}{suf}_kbK2_lj_s{s}"
            if not os.path.exists(f"{d}/predict_c05/metrics.json"):
                continue
            m = json.load(open(f"{d}/metrics.json")); p = json.load(open(f"{d}/predict_c05/metrics.json"))
            v.append([len(m["train_runs"]), m["heldout_chi2"], np.median([r["chi2"] for r in m["heldout_rows"]]),
                      np.mean([l2(r) for r in m["heldout_rows"]]), p["transfer"]["chi2"],
                      100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"])])
        if not v:
            print(f"{lab + ', fraction ' + frac:30s}  (not done)"); continue
        v = np.array(v); e = v.std(0, ddof=1) if len(v) > 1 else 0 * v[0]
        print(f"{lab + ', fraction ' + frac + f' ({len(v)})':30s}" + "".join(f"{a:8.2f}±{b:<5.2f}" for a, b in zip(v.mean(0), e)))
