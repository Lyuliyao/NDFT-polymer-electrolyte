"""SI table tab:S:lj (Lennard-Jones benchmark) from the LJ metrics: held-out chi2 (mean +- s.d., 3 seeds), profile error,
rho = 0.5 chi2 / profile, EL converged, quarter and half of the runs.  2026-10-04: spectrum-loss models (_kbK2).
    python make_tab_lj.py"""
import json, os, re, numpy as np
R = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj/runs/learn"; T = "c01_c02_c04_c06_c07"
L = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer/paper/output/latex/si/si_s9_compare.tex"
FORMS = [("Neural functional (19{,}641)", lambda f, s: f"joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus{f}_kbK2_lj_s{s}"),
         ("Neural functional, Gaussian kernels (18{,}841)", lambda f, s: f"joint_v2_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj_s{s}"),
         ("One-body network, $\\pm3\\sigma$ (32{,}514)", lambda f, s: f"joint_c1win_W30_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj_s{s}"),
         ("Lattice free energy, $R=1.5\\sigma$ (23{,}828)", lambda f, s: f"joint_cace_a5q3_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj_s{s}")]
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
def stats(name):
    m = json.load(open(f"{R}/{name}/metrics.json")); p = json.load(open(f"{R}/{name}/predict_c05/metrics.json"))
    ho = m["heldout_rows"]; el = sum(int(r["el"]["converged"]) if isinstance(r.get("el"), dict) else int(bool(r.get("el", 1))) for r in ho) + int(p["transfer"]["el_converged"]); nel = len(ho) + p["transfer"]["n_runs"]
    return dict(ho=float(np.mean([r["chi2"] for r in ho])), L2=float(np.mean([l2(r) for r in ho])), c5=p["transfer"]["chi2"], c5L2=100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"]), el=el, nel=nel)
def pm(v, nd):
    m, s = np.mean(v), np.std(v, ddof=1); return f"${m:.{nd}f}\\pm{s:.{nd}f}$"
rows = []
for lab, name in FORMS:
    full = [stats(name("", s)) for s in range(3)]; q = [stats(name("_f025", s)) for s in range(3)]; h = [stats(name("_f050", s)) for s in range(3)]
    ho = [x["ho"] for x in full]; nd = 1 if np.mean(ho) < 20 else 0
    rows.append(f"{lab} & {pm(ho, nd)} & {np.mean([x['L2'] for x in full]):.2f} & {np.mean([x['c5'] for x in full]):.{1 if np.mean([x['c5'] for x in full]) < 20 else 0}f} / {np.mean([x['c5L2'] for x in full]):.2f} & {full[0]['el']}/{full[0]['nel']} & "
                f"{np.mean([x['ho'] for x in q]):.{1 if np.mean([x['ho'] for x in q]) < 20 else 0}f} / {np.mean([x['L2'] for x in q]):.2f} & {np.mean([x['ho'] for x in h]):.{2 if np.mean([x['ho'] for x in h]) < 10 else 0}f} / {np.mean([x['L2'] for x in h]):.2f}\\\\")
v1 = stats(f"v1_{T}_kbK2")
rows.append(f"Pair closure (24) & {v1['ho']:.0f} & {v1['L2']:.1f} & {v1['c5']:.0f} / {v1['c5L2']:.1f} & {v1['el']}/{v1['nel']} & -- & --\\\\")
tex = open(L).read()
m = re.search(r"(\\label\{tab:S:lj\}.*?\\midrule\n)(.*?)(\\bottomrule)", tex, re.S); assert m
tex = tex[:m.start(2)] + "\n".join(rows) + "\n" + tex[m.start(3):]
open(L, "w").write(tex); print("\n".join(rows))
json.dump({lab: {"full": [stats(name("", s)) for s in range(3)], "f025": [stats(name("_f025", s)) for s in range(3)], "f050": [stats(name("_f050", s)) for s in range(3)]} for lab, name in FORMS} | {"pair": v1},
          open("/mnt/gs21/scratch/lyuliyao/salt_in_polymer/paper/figures/si/lj_kbK2_metrics.json", "w"), indent=1)
