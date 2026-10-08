"""Lennard-Jones benchmark with ONE species (2026-10-05, salt_in_polymer_lj1): the metrics of the Table 1 columns, in the format of
lj_kbK2_metrics.json (make_tab_lj.py), from the one-label models: held-out chi2, profile error, rho = 0.5 chi2 / profile, EL
converged, for the full data and the quarter and half fractions, three seeds; pair closure.  No SI table is written (the
consolidated SI has none).   python make_tab_lj1.py  -> paper/figures/si/lj1_kbK2_metrics.json"""
import json, os, numpy as np
R = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj1/runs/learn"; T = "c01_c02_c04_c06_c07"
FORMS = [("Neural functional (18{,}861)", lambda f, s: f"joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus{f}_kbK2_lj1_s{s}"),
         ("Neural functional, Gaussian kernels (17{,}801)", lambda f, s: f"joint_v2_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj1_s{s}"),
         ("One-body network, $\\pm3\\sigma$ (24{,}577)", lambda f, s: f"joint_c1win_W30_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj1_s{s}"),
         ("One-body network, $\\pm6\\sigma$ (32{,}257)", lambda f, s: f"joint_c1win_W60_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj1_s{s}"),
         ("Lattice free energy, $R=1.5\\sigma$ (19{,}426)", lambda f, s: f"joint_cace_a5q3_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj1_s{s}"),
         ("Lattice free energy, $R=3\\sigma$", lambda f, s: f"joint_cace_a5q6_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj1_s{s}")]
SP = ("cation", "anion")
l2 = lambda r: 100 * float(np.mean([r["profile"][s]["rel_l2"] for s in SP if s in r["profile"]]))
tl2 = lambda t: 100 * float(np.mean([t[f"{s}_rel_l2"] for s in SP if f"{s}_rel_l2" in t]))


def stats(name):
    d = f"{R}/{name}"
    if not os.path.exists(f"{d}/predict_c05/metrics.json"):
        return None
    m = json.load(open(f"{d}/metrics.json")); p = json.load(open(f"{d}/predict_c05/metrics.json")); ho = m["heldout_rows"]
    el = sum(int(r["el"]["converged"]) for r in ho) + int(p["transfer"]["el_converged"]); nel = len(ho) + p["transfer"]["n_runs"]
    return dict(ho=float(np.mean([r["chi2"] for r in ho])), ho_med=float(np.median([r["chi2"] for r in ho])), L2=float(np.mean([l2(r) for r in ho])),
                c5=p["transfer"]["chi2"], c5L2=tl2(p["transfer"]), el=el, nel=nel, val=m.get("val_chi2"), ntrain=len(m["train_runs"]))


out = {}
for lab, name in FORMS:
    out[lab] = {k: [x for x in (stats(name(suf, s)) for s in range(3)) if x] for k, suf in (("full", ""), ("f025", "_f025"), ("f050", "_f050"))}
    for k, v in out[lab].items():
        if v:
            print(f"{lab:48s} {k:5s} ({len(v)}) ho chi2 {np.mean([x['ho'] for x in v]):8.2f} (median {np.mean([x['ho_med'] for x in v]):6.2f})  L2 {np.mean([x['L2'] for x in v]):5.2f}%  "
                  f"rho.5 {np.mean([x['c5'] for x in v]):8.2f} / {np.mean([x['c5L2'] for x in v]):5.2f}%  EL {sum(x['el'] for x in v)}/{sum(x['nel'] for x in v)}  val {np.mean([x['val'] for x in v if x['val'] is not None]) if any(x['val'] is not None for x in v) else float('nan'):.2f}")
v1 = stats(f"v1_{T}_kbK2"); out["pair"] = v1
print(f"{'Pair closure (8)':48s} full      ho chi2 {v1['ho']:8.0f}  L2 {v1['L2']:5.1f}%  rho.5 {v1['c5']:8.0f} / {v1['c5L2']:5.1f}%  EL {v1['el']}/{v1['nel']}")
json.dump(out, open("/mnt/gs21/scratch/lyuliyao/salt_in_polymer/paper/figures/si/lj1_kbK2_metrics.json", "w"), indent=1)
