"""Size test: profile error and force residual of the trained functionals on boxes 1, 2, 4 and 8 times
longer along z (c = 0.04, both eps_r), from the prediction roots salt_in_polymer_bigbox/<box>/
(learn.protocol predict, nothing trained).  (A) p01 p09 p13 p15: the same potential in reduced units
at every size (m -> n m).  (B) p30 / p31: one box mode, k = k_1 / n, below the wavenumbers of the
training box (an extrapolation, reported separately).  eps_r = 2 at 1x: the production runs (training
runs for these tags) and the C7 control of p13 (never trained on).
    python figs/bigbox_summary.py
"""
import json, os, sys
PFX = sys.argv[1] if len(sys.argv) > 1 else "V2"      # V2: Gaussian form; NF: learned kernels; KBK2: the paper's model with the spectrum loss (2026-10-04)
PAIR = sys.argv[2] if len(sys.argv) > 2 else "V1"      # V1K: the pair closure refitted with the spectrum loss
import numpy as np
B = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox"
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])


def rows(root, model):
    f = f"{B}/{root}/runs/learn/{model}/predict_c004/metrics.json"
    if not os.path.exists(f):
        return {}
    return {r["tag"]: r for r in json.load(open(f))["transfer_rows"]}


out = {}
for e, boxes in (("7.5", [("x1", "eps75_c0.04_x1"), ("x2", "eps75_c0.04_x2"), ("x4", "eps75_c0.04_x4"), ("x8", "eps75_c0.04_x8")]),
                 ("2", [("x1 (training runs)", "eps2_c0.04_x1_prod"), ("x1 C7 control", "eps2_c0.04_x1_control"),
                        ("x2", "eps2_c0.04_x2"), ("x4", "eps2_c0.04_x4"), ("x8", "eps2_c0.04_x8")])):
    print(f"== eps_r = {e}: profile error % ({PFX} mean of 3 seeds [min-max] | V1), force chi2 (V2 mean | V1)")
    for fam, tags in (("(A) same potential", ("p01", "p09", "p13", "p15")), ("(B) k = k_1/n", ("p30", "p31")), ("(C) aperiodic", ("p60", "p61", "p62", "p63"))):
        print(f"  {fam}")
        for lab, root in boxes:
            v2 = [rows(root, f"{PFX}_s{s}") for s in (0, 1, 2)]; v1 = rows(root, PAIR)
            cells = []
            for t in tags:
                if not all(t in r for r in v2):
                    cells.append(f"{t}: {'-':>22s}"); continue
                L = [l2(r[t]) for r in v2]; C = [r[t]["chi2"] for r in v2]
                conv = all(r[t]["el"]["converged"] for r in v2)
                cells.append(f"{t}: {np.mean(L):5.2f} [{min(L):4.2f}-{max(L):4.2f}] | {l2(v1[t]) if t in v1 else float('nan'):5.2f}"
                             f"  chi2 {np.mean(C):5.2f} | {v1[t]['chi2'] if t in v1 else float('nan'):5.2f}{'' if conv else ' (EL!)'}")
                out.setdefault(e, {}).setdefault(lab, {})[t] = dict(V2_L2=L, V2_chi2=C, V1_L2=l2(v1[t]) if t in v1 else None,
                                                                      V1_chi2=v1[t]["chi2"] if t in v1 else None)
            if any("[" in c for c in cells):
                print(f"    {lab:20s}" + "   ".join(cells))
json.dump(out, open(f"{B}/bigbox_summary" + ("" if PFX == "V2" else "_" + PFX) + ".json", "w"), indent=1)


# --------------------------------------------------------------- driven-mode amplitudes
# sum over the driven modes and both ions of |a_pred - a_MD| / sum |a_MD|, a = 2 <n exp(-i k_m z)>:
# the slow undriven box modes, not equilibrated in the 4x and 8x boxes, do not enter.
import sys
sys.path.insert(0, "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/code")


def driven_error(root, tag, model):
    import learn.data as D
    os.environ["SIP_ROOT"] = f"{B}/{root}"
    sps, runs = D.load_all(root=f"{B}/{root}")
    r = next((x for x in runs if x.tag == tag), None)
    f = f"{B}/{root}/runs/learn/{model}/predict_c004/predicted.npz"
    if r is None or not os.path.exists(f):
        return None
    p = np.load(f)
    key = f"transfer_c0.04_{tag}"
    if key not in p.files:
        return None
    npred = p[key]
    terms = r.spec.get("neutral", []) + r.spec.get("charged", [])
    amax = max(t["A"] for t in terms)
    ms = sorted({t["m"] for t in terms if t["A"] >= 1e-3 * amax})
    N = r.n.shape[1]; z = (np.arange(N) + 0.5) * float(json.load(open(f"{B}/{root}/runs/{root}/zero_field/D.json"))["lz"]) / N
    num = den = 0.0
    for m in ms:
        e = np.exp(-2j * np.pi * m * z / z[-1] * (N - 0.5) / N)
        for a in range(2):
            am, ap = 2 * (r.n[a] * e).mean(), 2 * (npred[a] * e).mean()
            num += abs(ap - am); den += abs(am)
    return 100 * num / den


print(f"\nDriven-mode amplitude error % ({PFX} mean of 3 seeds [min-max] | V1); * = undriven box modes not equilibrated")
flag = {}
import csv
for r in csv.DictReader(open("/mnt/gs21/scratch/lyuliyao/salt_in_polymer/bigbox/field_analysis/acceptance_summary.csv")):
    flag[(os.path.basename(r["state"].rstrip("/")), r["tag"])] = r["verdict"]
for e, boxes in (("7.5", [("x1", "eps75_c0.04_x1"), ("x2", "eps75_c0.04_x2"), ("x4", "eps75_c0.04_x4"), ("x8", "eps75_c0.04_x8")]),
                 ("2", [("x1 (training runs)", "eps2_c0.04_x1_prod"), ("x1 C7 control", "eps2_c0.04_x1_control"),
                        ("x2", "eps2_c0.04_x2"), ("x4", "eps2_c0.04_x4"), ("x8", "eps2_c0.04_x8")])):
    print(f"== eps_r = {e}")
    for lab, root in boxes:
        cells = []
        for t in ("p01", "p09", "p13", "p15", "p30", "p31", "p60", "p61", "p62", "p63"):
            v2 = [driven_error(root, t, f"{PFX}_s{s}") for s in (0, 1, 2)]
            if any(v is None for v in v2):
                continue
            v1 = driven_error(root, t, PAIR)
            star = "*" if "slow" in flag.get((root, t), "") else " "
            cells.append(f"{t}{star} {np.mean(v2):5.1f} [{min(v2):4.1f}-{max(v2):4.1f}] | {v1 if v1 is not None else float('nan'):5.1f}")
            out.setdefault(e, {}).setdefault(lab, {}).setdefault(t, {}).update(V2_driven=v2, V1_driven=v1)
        if cells:
            print(f"    {lab:20s}" + "   ".join(cells))
json.dump(out, open(f"{B}/bigbox_summary" + ("" if PFX == "V2" else "_" + PFX) + ".json", "w"), indent=1)
