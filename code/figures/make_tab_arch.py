"""SI tables of section S9 and the rows of Table 1 of the main text, from the retrained models (2026-10-04, spectrum loss):
tab:S:arch (accuracy of the forms, both eps_r), tab:S:archfrac (training-set fractions), tab:S:archstruct (structural checks,
max (mean) from the recomputed per-run values), Table 1 rows.  Sources: arch_summary_kn.json of each eps_r root
(figs/arch_summary_kn.py), the metrics of the fraction models, structure_eps*.jsonl (recompute_structure.py), lj_kbK2_metrics.json.
    python make_tab_arch.py"""
import json, os, re, numpy as np
R = "/mnt/research/MultiscaleML_group/Liyao"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"; T = "c001_c002_c004_c006_c008"
L = f"{S}/paper/output/latex/"; SUM = f"{S}/paper/figures/summaries_kbK2"
A = {"7.5": json.load(open(f"{R}/salt_in_polymer_eps75/runs/learn/interpretation/arch_summary_kn.json")), "2": json.load(open(f"{R}/salt_in_polymer_eps2/runs/learn/interpretation/arch_summary_kn.json"))}
ST = {}
for e, f in (("7.5", "structure_eps75.jsonl"), ("2", "structure_eps2.jsonl")):
    ST[e] = {}
    for ln in open(f"{SUM}/{f}"):
        d = json.loads(ln); ST[e].setdefault(d["label"], []).append(d["stats"])
LAB = [("Neural functional (this work)", "ours (learned kernels)", "19{,}641"), ("Neural functional, Gaussian kernels", "ours, Gaussian basis", "18{,}841"),
       ("One-body network, $\\pm3\\sigma$", "Sammuller c1, +-3 sigma", "32{,}514"), ("One-body network, $\\pm6\\sigma$", "Sammuller c1, +-6 sigma", "47{,}874"),
       ("Lattice free energy, $R=1.5\\sigma$", "Cheng CACE, R 1.5 sigma", "23{,}828"), ("Lattice free energy, $R=3\\sigma$", "Cheng CACE, R 3 sigma", "23{,}828")]
pm = lambda v, nd=2: f"${np.mean(v):.{nd}f}\\pm{np.std(v, ddof=1):.{nd}f}$" if len(v) > 1 else f"{np.mean(v):.{nd}f}"
sci = lambda x: ("$10^{%d}$" % round(np.log10(x))) if x > 0 and abs(np.log10(x) - round(np.log10(x))) < 0.25 else (f"${x / 10 ** np.floor(np.log10(x)):.0f}\\times10^{{{int(np.floor(np.log10(x)))}}}$" if x > 0 else "0")
# ---- tab:S:arch
rows = []
for name, key, npar in LAB:
    for e in ("7.5", "2"):
        rs = A[e][key]
        p27 = f"{pm([r['p27'] for r in rs], 1)}" if e == "2" and all("p27" in r for r in rs) else "--"
        rows.append(f"{name if e == '7.5' else ''} & {e} & {npar if e == '7.5' else ''} & {pm([r['ho'] for r in rs])} & {pm([r['L2'] for r in rs])} & {pm([r['c5'] for r in rs])} & {pm([r['c5L2'] for r in rs])} & {pm([r['Sk5'] for r in rs], 1)} & {sum(r['el'] for r in rs)}/{sum(r['nel'] for r in rs)} & {p27}\\\\")
for e in ("7.5", "2"):
    r = A[e]["V1 (pair closure)"][0]
    rows.append(f"{'Pair closure' if e == '7.5' else ''} & {e} & {'24' if e == '7.5' else ''} & {r['ho']:.2f} & {r['L2']:.2f} & {r['c5']:.2f} & {r['c5L2']:.2f} & {r['Sk5']:.1f} & {r['el']}/{r['nel']} & {(f'{r[chr(112)+chr(50)+chr(55)]:.1f}' if 'p27' in r else '--') if e == '2' else '--'}\\\\")
# ---- tab:S:archfrac (held-out chi2 / c=0.05 chi2, three-seed means) from the fraction models
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
def fstats(e, name):
    root = f"{R}/salt_in_polymer_{'eps75' if e == '7.5' else 'eps2'}/runs/learn"; m = json.load(open(f"{root}/{name}/metrics.json")); p = json.load(open(f"{root}/{name}/predict_c005/metrics.json"))
    ho = [r for r in m["heldout_rows"] if not (e == "2" and r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9)]
    return dict(ho=float(np.mean([r["chi2"] for r in ho])), c5=p["transfer"]["chi2"], L2=float(np.mean([l2(r) for r in ho])), ntr=len(m["train_runs"]))
FR = {"7.5": {}, "2": {}}
for e in ("7.5", "2"):
    tag = "long" if e == "7.5" else "full"; tagk = "_long" if e == "7.5" else ""
    for frac, suf in (("0.25", "_f025"), ("0.5", "_f050"), ("1", "")):
        FR[e][frac] = {"neural functional": [fstats(e, f"joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus{suf}_kbK2{tagk}_s{s}") for s in range(3)],
                       "Gaussian kernels": [fstats(e, f"joint_v2_L1_C4_R128x128_H64_{T}_val3{suf}_kbK2_{tag}_s{s}") for s in range(3)],
                       "one-body network": [fstats(e, f"joint_c1win_W30_L1_C4_R128x128_H64_{T}_val3{suf}_kbK2_{tag}_s{s}") for s in range(3)],
                       "lattice free energy": [fstats(e, f"joint_cace_a5q3_L1_C4_R128x128_H64_{T}_val3{suf}_kbK2_{tag}_s{s}") for s in range(3)]}
frows = []
for e in ("7.5", "2"):
    for frac in ("0.25", "0.5", "1"):
        d = FR[e][frac]; ntr = np.mean([x["ntr"] for x in d["neural functional"]])
        cell = lambda k: f"{np.mean([x['ho'] for x in d[k]]):.2f} / {np.mean([x['c5'] for x in d[k]]):.2f}" if np.mean([x['ho'] for x in d[k]]) < 10 else f"{np.mean([x['ho'] for x in d[k]]):.1f} / {np.mean([x['c5'] for x in d[k]]):.1f}"
        frows.append(f"{e} & {frac} ({ntr:.1f}) & {cell('neural functional')} & {cell('Gaussian kernels')} & {cell('one-body network')} & {cell('lattice free energy')}\\\\")
qnote = ("Profile errors with a quarter of the runs: neural functional " + " and ".join(f"{np.mean([x['L2'] for x in FR[e]['0.25']['neural functional']]):.1f}\\% ($\\varepsilon_r={e}$)" for e in ("7.5", "2"))
         + "; Gaussian kernels " + " and ".join(f"{np.mean([x['L2'] for x in FR[e]['0.25']['Gaussian kernels']]):.1f}\\%" for e in ("7.5", "2"))
         + "; one-body network " + " and ".join(f"{np.mean([x['L2'] for x in FR[e]['0.25']['one-body network']]):.1f}\\%" for e in ("7.5", "2")) + ".")
# ---- tab:S:archstruct: max (mean) over runs and seeds, from the recomputed per-run values; W asym, refl, loop, SL from arch_summary_kn
srows = []
for name, key, npar in LAB:
    for e in ("7.5", "2"):
        st = ST[e][key]; rs = A[e][key]
        noe = np.concatenate([s["noether_all"] for s in st]); ja = np.concatenate([s["jac_asym_all"] for s in st]); lp = np.concatenate([s["loop_all"] for s in st])
        jm = f"{np.mean(ja):.2f}" if np.mean(ja) > 1e-3 else sci(np.mean(ja)); nm = f"{np.mean(noe):.3f}" if np.mean(noe) > 1e-3 else sci(np.mean(noe))
        srows.append(f"{name if e == '7.5' else ''} & {e} & {sci(noe.max())} ({nm}) & {sci(ja.max())} ({jm}) & {sci(max(r['W_asym_max'] for r in rs))} & {sci(max(r['refl_max'] for r in rs))} & {sci(lp.max())} & {sci(max(r['SL_max'] for r in rs))}\\\\")
tex = open(L + "si/si_tab_arch.tex").read()
for lab, body in (("tab:S:arch", rows), ("tab:S:archfrac", frows), ("tab:S:archstruct", srows)):
    m = re.search(r"(\\label\{" + re.escape(lab) + r"\}.*?\\midrule\n)(.*?)(\\bottomrule)", tex, re.S); assert m, lab
    tex = tex[:m.start(2)] + "\n".join(body) + "\n" + tex[m.start(3):]
m = re.search(r"(\\addtabletext\{)Profile errors with a quarter of the runs:.*?one-body network [^.]*\.", tex, re.S); assert m, "qnote"
tex = tex[:m.start()] + m.group(1) + qnote + tex[m.end():]
open(L + "si/si_tab_arch.tex", "w").write(tex); print("si_tab_arch.tex regenerated"); print("\n".join(rows + frows + srows))
# ---- Table 1 rows of the main text
LJ = json.load(open(f"{S}/paper/figures/si/lj_kbK2_metrics.json")); D = json.load(open(f"{R}/salt_in_polymer_eps2/runs/learn/interpretation/dijkman_summary.json")) if os.path.exists(f"{R}/salt_in_polymer_eps2/runs/learn/interpretation/dijkman_summary.json") else None
def t1(name, key, npar, ljkey, c1=False):
    a75, a2 = A["7.5"][key], A["2"][key]; q75, q2 = FR["7.5"]["0.25"][c1], FR["2"]["0.25"][c1]
    lj = LJ[ljkey]["full"]; st75, st2 = ST["7.5"][key], ST["2"][key]
    noe = np.concatenate([s["noether_all"] for s in st75 + st2]); ja = np.concatenate([s["jac_asym_all"] for s in st75 + st2]); lp = np.concatenate([s["loop_all"] for s in st75 + st2])
    f = lambda x, nd=2: f"{x:.{nd}f}" if x < 10 else f"{x:.0f}"
    struct = f"{sci(noe.max())} ({sci(np.mean(noe)) if np.mean(noe) < 1e-3 else f'{np.mean(noe):.3f}'}); {sci(ja.max())} ({sci(np.mean(ja)) if np.mean(ja) < 1e-3 else f'{np.mean(ja):.2f}'}); {sci(lp.max())} ({sci(np.mean(lp)) if np.mean(lp) < 1e-3 else f'{np.mean(lp):.3f}'})"
    return (f"{name} & {npar} & {f(np.mean([x['ho'] for x in lj]), 1)} & {np.mean([x['L2'] for x in lj]):.2f} & {f(np.mean([r['ho'] for r in a75]))} & {f(np.mean([r['ho'] for r in a2]))} & {f(np.mean([r['c5'] for r in a75]))} & {f(np.mean([r['c5'] for r in a2]))} & "
            f"{f(np.mean([x['ho'] for x in q75]))} & {f(np.mean([x['ho'] for x in q2]))} & {f(np.mean([r['p27'] for r in a2]), 1)} & {struct}\\\\")
t1rows = [t1("Neural functional (this work)", "ours (learned kernels)", "19{,}641", "Neural functional (19{,}641)", "neural functional"),
          t1("One-body network, $\\pm3\\sigma$", "Sammuller c1, +-3 sigma", "32{,}514", "One-body network, $\\pm3\\sigma$ (32{,}514)", "one-body network"),
          t1("Lattice free energy$^{\\dagger}$", "Cheng CACE, R 1.5 sigma", "23{,}828", "Lattice free energy, $R=1.5\\sigma$ (23{,}828)", "lattice free energy")]
v75, v2 = A["7.5"]["V1 (pair closure)"][0], A["2"]["V1 (pair closure)"][0]; vlj = LJ["pair"]
t1rows.append(f"Pair closure & 24 & {vlj['ho']:.0f} & {vlj['L2']:.1f} & {v75['ho']:.2f} & {v2['ho']:.1f} & {v75['c5']:.2f} & {v2['c5']:.2f} & -- & -- & {v2['p27']:.0f} & exact\\\\")
print("== Table 1 rows (convolutional row from make_tab_dijkman / dijkman_summary):"); print("\n".join(t1rows))
json.dump({"table1": t1rows}, open(f"{S}/paper/figures/si/table1_rows.json", "w"), indent=1)
