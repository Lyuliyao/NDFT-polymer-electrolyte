"""Single-model tables (2026-10-06; user: report one trained model per form, the best one, no seed statistics).
Every form, data fraction and dielectric constant is represented by the initialization with the lowest validation
residual (val_chi2 of metrics.json); for the paper's functional this is seed 0 at eps_r = 7.5 and seed 1 at eps_r = 2
(figstyle.SEED).  Splices the table bodies into the consolidated SI (si_s2_methods.tex: tab:S:spectrum;
si_s3_results.tex: tab:S:profiles, tab:S:sk; si_s4_validation.tex: tab:S:bigbox, tab:S:field2d, tab:S:archfrac,
tab:S:archstruct, tab:S:dijkman) and the Table 1 rows of main.tex, and writes the selection and the numbers quoted in
the text to single_model_metrics.json.   Run after make_fig2.py, make_fig3.py, kernel_k1.py, make_figS_longmode.py.
    python make_tab_single.py"""
import json, os, re
import numpy as np
R = "/mnt/research/MultiscaleML_group/Liyao"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
LX = os.environ.get("NDFT_LATEX", f"{S}/paper/output/latex"); FG = f"{S}/paper/figures"; T = "c001_c002_c004_c006_c008"
# NDFT_LATEX: the LaTeX tree to write into (2026-10-07: the Overleaf repository CDFT-electrolytes/v1, same file layout)
ROOT = {"7.5": f"{R}/salt_in_polymer_eps75/runs/learn", "2": f"{R}/salt_in_polymer_eps2/runs/learn"}
TAG = {"7.5": "long", "2": "full"}; TAGK = {"7.5": "_long", "2": ""}
PAIR = {"7.5": f"joint_v1_L1_C4_R128x128_H64_{T}_val3_kn1softplus_kbK2_long_s2", "2": f"joint_v1_L1_C4_R128x128_H64_{T}_val3_kn1softplus_kbK2_s0"}   # 2026-10-08: pair closure with the learned kernels
EPS = ("7.5", "2"); CS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]


def pattern(form, e, frac=""):
    return {"ours": f"joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus{frac}_kbK2{TAGK[e]}_s{{s}}",
            "gauss": f"joint_v2_L1_C4_R128x128_H64_{T}_val3{frac}_kbK2_{TAG[e]}_s{{s}}",
            "c1win": f"joint_c1win_W30_L1_C4_R128x128_H64_{T}_val3{frac}_kbK2_{TAG[e]}_s{{s}}",
            "c1win6": f"joint_c1win_W60_L1_C4_R128x128_H64_{T}_val3{frac}_kbK2_{TAG[e]}_s{{s}}",
            "cace": f"joint_cace_a5q3_L1_C4_R128x128_H64_{T}_val3{frac}_kbK2_{TAG[e]}_s{{s}}",
            "cace6": f"joint_cace_a5q6_L1_C4_R128x128_H64_{T}_val3{frac}_kbK2_{TAG[e]}_s{{s}}",
            "cnn": f"joint_cnn_d2_L1_C4_{T}_val3_force_kbK2{TAGK[e]}_s{{s}}",
            "plain": f"joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus{TAGK[e]}_s{{s}}"}[form]


def metrics(e, n):
    return json.load(open(f"{ROOT[e]}/{n}/metrics.json"))


SEL = {}
def best(form, e, frac=""):
    pat = pattern(form, e, frac); vs = [metrics(e, pat.format(s=s)).get("val_chi2", np.inf) for s in range(3)]
    s = int(np.argmin(vs)); SEL[f"{form}{frac} {e}"] = dict(seed=s, val=[round(v, 4) for v in vs], model=pat.format(s=s)); return s


SEED = {e: best("ours", e) for e in EPS}
assert SEED == {"7.5": 0, "2": 1}, SEED            # = figstyle.SEED, used by the figures
isp27 = lambda e, r: e == "2" and r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9
def sci(x):
    """one significant digit: 0.14, 0.006, 2x10^-6 (decimals down to 1e-3, powers of ten below)"""
    if x <= 0:
        return "0"
    if x >= 0.0095:
        return f"{x:.2f}"
    if x >= 1e-3:
        return f"{x:.3f}"
    e = int(np.floor(np.log10(x))); m = int(round(x / 10 ** e))
    if m == 10:
        m, e = 1, e + 1
    return f"$10^{{{e}}}$" if m == 1 else f"${m}\\times10^{{{e}}}$"
f2 = lambda x: f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")
OUT = {"selection": SEL}


def splice(path, label, rows):
    tex = open(path).read()
    m = re.search(r"(\\label\{" + re.escape(label) + r"\}.*?\\midrule\n)(.*?)(\\bottomrule)", tex, re.S); assert m, label
    tex = tex[:m.start(2)] + "\n".join(rows) + "\n" + tex[m.start(3):]; open(path, "w").write(tex)


# ------------------------------------------------------------------ data
A = {e: json.load(open(f"{ROOT[e]}/interpretation/arch_summary_kn.json")) for e in EPS}
LAB = {"ours": "ours (learned kernels)", "gauss": "ours, Gaussian basis", "c1win": "Sammuller c1, +-3 sigma",
       "c1win6": "Sammuller c1, +-6 sigma", "cace": "Cheng CACE, R 1.5 sigma", "cace6": "Cheng CACE, R 3 sigma"}
for e in EPS:
    for k in LAB.values():
        assert len(A[e][k]) == 3, (e, k)
ST = {e: {} for e in EPS}
for e, f in (("7.5", "structure_eps75.jsonl"), ("2", "structure_eps2.jsonl")):
    for ln in open(f"{FG}/summaries_kbK2/{f}"):
        d = json.loads(ln); ST[e][d["model"]] = d["stats"]
DJ = json.load(open(f"{ROOT['2']}/interpretation/dijkman_summary.json"))


def arch(form, e):
    s = best(form, e); return A[e][LAB[form]][s], ST[e].get(pattern(form, e).format(s=s))


def heldout(e, n):
    m = metrics(e, n); p = json.load(open(f"{ROOT[e]}/{n}/predict_c005/metrics.json"))
    ho = [r for r in m["heldout_rows"] if not isp27(e, r)]
    return dict(ho=float(np.mean([r["chi2"] for r in ho])), c5=p["transfer"]["chi2"])


# ------------------------------------------------------------------ Table 1 (main text)
LJ = json.load(open(f"{FG}/si/lj1_kbK2_metrics.json"))
def ljbest(key, frac="full"):
    v = LJ[key][frac]; s = int(np.argmin([x["val"] for x in v])); SEL[f"LJ {key} {frac}"] = dict(seed=s, val=[x["val"] for x in v]); return v[s]
LJK = {"ours": "Neural functional (18{,}861)", "c1win": "One-body network, $\\pm3\\sigma$ (24{,}577)", "cace": "Lattice free energy, $R=1.5\\sigma$ (19{,}426)"}


def struct_cell(stats_list, lab):
    noe = np.concatenate([s["noether_all"] for s in stats_list]); ja = np.concatenate([s["jac_asym_all"] for s in stats_list]); lp = np.concatenate([s["loop_all"] for s in stats_list])
    mean = lambda x, nd: sci(float(np.mean(x)))
    cell = f"{sci(noe.max())} ({mean(noe, 3)}); {sci(ja.max())}" + (f" ({mean(ja, 2)})" if np.mean(ja) > 1e-3 else "") + f"; {sci(lp.max())}" + (f" ({mean(lp, 3)})" if np.mean(lp) > 1e-3 else "")
    OUT.setdefault("structure", {})[lab] = dict(noether_max=float(noe.max()), noether_mean=float(noe.mean()), jac_max=float(ja.max()), jac_mean=float(ja.mean()), loop_max=float(lp.max()), loop_mean=float(lp.mean()))
    return cell


t1 = []
for form, lab, npar in (("ours", "Neural functional (this work)", "19{,}641"), ("c1win", "One-body network, $\\pm3\\sigma$", "32{,}514"), ("cace", "Lattice free energy$^{\\dagger}$", "23{,}828")):
    a = {e: arch(form, e) for e in EPS}; q = {e: heldout(e, pattern(form, e, "_f025").format(s=best(form, e, "_f025"))) for e in EPS}
    lj = ljbest(LJK[form]); p27 = a["2"][0]["p27"]
    t1.append(f"{lab} & {npar} & {(f'{lj[chr(104)+chr(111)]:.1f}' if lj['ho'] < 100 else f'{lj[chr(104)+chr(111)]:.0f}')} & {lj['L2']:.2f} & {f2(a['7.5'][0]['ho'])} & {f2(a['2'][0]['ho'])} & {f2(a['7.5'][0]['c5'])} & {f2(a['2'][0]['c5'])} & "
              f"{f2(q['7.5']['ho'])} & {f2(q['2']['ho'])} & {p27:.1f} & {struct_cell([a['7.5'][1], a['2'][1]], form)}\\\\")
    OUT.setdefault("table1", {})[form] = dict(lj=lj, ho={e: a[e][0]["ho"] for e in EPS}, c5={e: a[e][0]["c5"] for e in EPS}, quarter=q, p27=p27,
                                               L2={e: a[e][0]["L2"] for e in EPS}, Sk5={e: a[e][0]["Sk5"] for e in EPS}, el={e: [a[e][0]["el"], a[e][0]["nel"]] for e in EPS})
# convolutional free energy (force matching), the seed with the lowest validation residual
cnn = {e: DJ[e]["CNN, force"][best("cnn", e)] for e in EPS}
noe = np.concatenate([cnn[e]["noether_all"] for e in EPS])
t1.append(f"Convolutional free energy$^{{\\ddagger}}$ & 24{{,}}321 & -- & -- & {cnn['7.5']['ho_chi2']:.0f} & {cnn['2']['ho_chi2']:.0f} & {cnn['7.5']['c5_chi2']:.0f} & {cnn['2']['c5_chi2']:.0f} & -- & -- & "
          f"{cnn['2']['p27'][0]:.0f} & {sci(noe.max())} ({sci(float(noe.mean()))}); $10^{{-16}}$; $10^{{-15}}$\\\\")
OUT["table1"]["cnn"] = {e: dict(ho=cnn[e]["ho_chi2"], c5=cnn[e]["c5_chi2"], el=[cnn[e]["el"], cnn[e]["nel"]], L2=cnn[e]["ho_L2"]) for e in EPS} | {"p27": cnn["2"]["p27"][0]}
v = {e: A[e]["V1 (pair closure)"][0] for e in EPS}; vlj = LJ["pair"]
t1.append(f"Pair closure & 24 & {vlj['ho']:.0f} & {vlj['L2']:.1f} & {v['7.5']['ho']:.2f} & {v['2']['ho']:.1f} & {v['7.5']['c5']:.2f} & {v['2']['c5']:.2f} & -- & -- & {v['2']['p27']:.0f} & exact\\\\")
# 2026-10-07: Table 1 is written in profile errors by make_tab_profile.py; the chi^2 rows above are kept in the metrics only
print("== Table 1 (chi2, not written)"); print("\n".join(t1))

# ------------------------------------------------------------------ tab:S:archfrac, tab:S:archstruct
rows = []
for e in EPS:
    for frac, suf in (("0.25", "_f025"), ("0.5", "_f050"), ("1", "")):
        cells = []
        for form in ("ours", "c1win", "cace"):
            h = heldout(e, pattern(form, e, suf).format(s=best(form, e, suf)))
            cells.append(f"{f2(h['ho'])} / {f2(h['c5'])}"); OUT.setdefault("archfrac", {})[f"{form} {e} {frac}"] = h
        rows.append(f"{e} & {frac} & " + " & ".join(cells) + "\\\\")
print("== archfrac (chi2, not written; make_tab_profile.py writes profile errors)"); print("\n".join(rows))
rows = []
for form, lab in (("ours", "Neural functional"), ("c1win", "One-body network"), ("cace", "Lattice free energy")):
    for e in EPS:
        a, st = arch(form, e)
        noe, ja, lp = np.array(st["noether_all"]), np.array(st["jac_asym_all"]), np.array(st["loop_all"])
        nm = sci(float(noe.mean())); jm = sci(float(ja.mean()))
        rows.append(f"{lab} & {e} & {sci(noe.max())} ({nm}) & {sci(ja.max())} ({jm}) & {sci(a['refl_max'])} & {sci(lp.max())}\\\\")
splice(f"{LX}/si/si_s4_validation.tex", "tab:S:archstruct", rows); print("== archstruct"); print("\n".join(rows))

# ------------------------------------------------------------------ tab:S:dijkman
pct = lambda x: f"{x:.0f}" if x >= 10 else f"{x:.1f}"
rows = []
for e in EPS:
    for lab, d, mod in (("neural functional", dict(DJ[e]["ours, force"][SEED[e]]), pattern("ours", e).format(s=SEED[e])), ("convolutional", dict(cnn[e]), SEL[f"cnn {e}"]["model"])):
        mm = metrics(e, mod); d["ho_chi2"] = float(mm["heldout_chi2"])
        d["ho_L2"] = float(np.mean([100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"]) for r in mm["heldout_rows"]]))
        rows.append(f"{e} & {lab} & {f2(d['ho_chi2'])} & {f2(d['ho_L2'])} & {d['el']}/{d['nel']} & {pct(d['Sk_ZZ'])} / {pct(d['Sk_NN'])}\\\\")
splice(f"{LX}/si/si_s4_validation.tex", "tab:S:dijkman", rows); print("== dijkman"); print("\n".join(rows))

# ------------------------------------------------------------------ tab:S:spectrum
KB = json.load(open(f"{ROOT['2']}/interpretation/kboost_eval.json")); REL = json.load(open(f"{R}/salt_in_polymer_bigbox/bigbox_longmode_relax.json"))
rows = []
for e, key in (("7.5", "eps75"), ("2", "eps2")):
    for lab, kbk, mkey, s in (("without", "kbkb0", "model_NF_cation", best("plain", e)), ("with", "kbkbK2", "model_kbK2_cation", SEED[e])):
        k = dict(KB[f"{e}/{kbk}/s{s}"]); mm = metrics(e, pattern("plain" if kbk == "kbkb0" else "ours", e).format(s=s))
        k["ho_chi2"] = float(mm["heldout_chi2"]); k["ho_L2"] = float(np.mean([100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"]) for r in mm["heldout_rows"]]))
        dfc = lambda nb, tag: 100 * (1 - REL[f"{key} x{nb} {tag}"][mkey][s] / REL[f"{key} x{nb} {tag}"]["cat"]["A"])
        rows.append(f"{e} & {lab} & {k['ratio']['0.04']:.2f} & {k['Gamma_k1']['0.04']:.2f} & {k['ho_L2']:.2f} & "   # 2026-10-08: no chi2 column (the user's table layout)
                    f"{dfc(2, 'p30'):.0f} / {dfc(2, 'p31'):.0f} & {dfc(4, 'p30'):.0f} / {dfc(4, 'p31'):.0f}\\\\")
        OUT.setdefault("spectrum", {})[f"{e} {lab}"] = dict(seed=s, ratio=k["ratio"]["0.04"], Gamma=k["Gamma_k1"]["0.04"], ho=k["ho_chi2"], L2=k["ho_L2"],
                                                           k1_2=[dfc(2, "p30"), dfc(2, "p31")], k1_4=[dfc(4, "p30"), dfc(4, "p31")])
splice(f"{LX}/si/si_s2_methods.tex", "tab:S:spectrum", rows); print("== spectrum"); print("\n".join(rows))

# ------------------------------------------------------------------ tab:S:profiles, tab:S:sk
rows = []
for e in EPS:
    m = metrics(e, pattern("ours", e).format(s=SEED[e])); p5 = json.load(open(f"{ROOT[e]}/{pattern('ours', e).format(s=SEED[e])}/predict_c005/metrics.json"))
    pm = metrics(e, PAIR[e]); pp5 = json.load(open(f"{ROOT[e]}/{PAIR[e]}/predict_c005/metrics.json"))
    def cell(rs):
        return f"{np.mean([r['chi2'] for r in rs]):.3f}", f"{100 * np.mean([r['profile']['cation']['rel_l2'] for r in rs]):.1f} / {100 * np.mean([r['profile']['anion']['rel_l2'] for r in rs]):.1f}"
    def pcell(rs):
        return f"{np.mean([r['chi2'] for r in rs]):.2f}", f"{100 * np.mean([r['profile']['cation']['rel_l2'] for r in rs]):.1f} / {100 * np.mean([r['profile']['anion']['rel_l2'] for r in rs]):.1f}"
    for c in CS:
        if c == 0.05:
            t, tp = p5["transfer"], pp5["transfer"]
            rows.append(f"{e} & 0.05 & {t['chi2']:.3f} & {100 * t['cation_rel_l2']:.1f} / {100 * t['anion_rel_l2']:.1f} & {tp['chi2']:.2f} & {100 * tp['cation_rel_l2']:.1f} / {100 * tp['anion_rel_l2']:.1f}\\\\"); continue
        sel = lambda rows_, p27: [r for r in rows_ if abs(r["conc"] - c) < 1e-9 and (p27 is None or isp27(e, r) == p27)]
        a, b = cell(sel(m["heldout_rows"], None)); x, y = pcell(sel(pm["heldout_rows"], None)); rows.append(f"{e} & {c:g} & {a} & {b} & {x} & {y}\\\\")
splice(f"{LX}/si/si_s3_results.tex", "tab:S:profiles", rows); print("== profiles"); print("\n".join(rows))
G = json.load(open(f"{FG}/fig3_metrics.json"))["Gamma"]
SK = {"7.5": json.load(open(f"{FG}/structure_metrics.json")), "2": json.load(open(f"{FG}/si/structure_eps2_metrics.json"))}
rows = []
for e in EPS:
    fun = SK[e][f"fun_s{SEED[e]}"]; pair = SK[e]["pair"]; g = G[e]
    for c in CS:
        k = f"{c:g}"; md = g["MD"][k]
        rows.append(f"{e} & {k} & {100 * fun[k]['ZZ']:.1f} / {100 * pair[k]['ZZ']:.1f} & {100 * fun[k]['NN']:.1f} / {100 * pair[k]['NN']:.1f} & ${md[0]:.2f}\\pm{md[1]:.2f}$ & {g['fun_seeds'][0][k]:.2f} & {g['pair'][k]:.2f}\\\\")
    OUT.setdefault("Sk", {})[e] = {f"{c:g}": dict(ZZ=fun[f"{c:g}"]["ZZ"], NN=fun[f"{c:g}"]["NN"], Gamma=g["fun_seeds"][0][f"{c:g}"]) for c in CS}
splice(f"{LX}/si/si_s3_results.tex", "tab:S:sk", rows); print("== sk"); print("\n".join(rows))

# ------------------------------------------------------------------ tab:S:bigbox, tab:S:field2d
B = json.load(open(f"{R}/salt_in_polymer_bigbox/bigbox_summary_KBK2.json"))
rows = []; allv = {}
for e in EPS:
    for i, (nb, key) in enumerate(((1, "x1" if e == "7.5" else "x1 (training runs)"), (2, "x2"), (4, "x4"), (8, "x8"))):
        cells = []
        for t in ("p01", "p09", "p13", "p15"):
            r = B[e][key][t]; vv = r["V2_driven"][SEED[e]]; allv.setdefault(e, []).append(vv); cells.append(f"{vv:.1f} / {'--' if r['V1_driven'] is None else f'{r[chr(86)+chr(49)+chr(95)+chr(100)+chr(114)+chr(105)+chr(118)+chr(101)+chr(110)]:.1f}'}")   # 2026-10-08: the pair closure is predicted only in the 2x and 4x boxes
        rows.append(f"{e if i == 0 else ''} & {nb} & " + " & ".join(cells) + "\\\\")
print("== bigbox (repeated potentials; removed from the paper 2026-10-07, not written)"); print("\n".join(rows))
OUT["bigbox_driven_range"] = {e: [min(v), max(v)] for e, v in allv.items()}
OUT["bigbox_aperiodic"] = {e: {t: dict(L2=B[e]["x2"][t]["V2_L2"][SEED[e]], chi2=B[e]["x2"][t]["V2_chi2"][SEED[e]], driven=B[e]["x2"][t].get("V2_driven", [None] * 3)[SEED[e]],
                                       pair_L2=B[e]["x2"][t]["V1_L2"]) for t in ("p60", "p61", "p62", "p63") if t in B[e]["x2"]} for e in EPS}
# 2026-10-07 (user): the 2D table reports profile errors for every example, with the MD noise as the reference, instead of
# the rms deviation in MD standard errors.  figs/field2d_profile_errors.py: relative L2 error of the map on the Fourier modes
# that MD resolves (|a| > 3 s.e.), cations and anions averaged, and the MD noise level of the same quantity.
PE = json.load(open(f"{R}/salt_in_polymer_field2d/field2d_profile_errors.json"))
POT = (("p50", "egg carton"), ("p51", "neutral rods"), ("p52", "charged rods"), ("p53", "checkerboard"), ("p54", "aperiodic I"), ("p55", "aperiodic II"))
p1 = lambda x: f"{x:.0f}" if x >= 10 else f"{x:.1f}"
rows = []
for e, sysn in (("7.5", "eps75"), ("2", "eps2")):
    for i, (tag, lab) in enumerate(POT):
        r = PE[f"{sysn}/{tag}"]
        rows.append(f"{e if i == 0 else ''} & {lab} & {sum(r['modes_resolved'])} & {r['neural functional']['res']:.1f} & {r['pair closure']['res']:.1f} & "
                    f"{p1(r['Poisson-Boltzmann']['res'])} & {r['noise']['res']:.1f}\\\\")
        OUT.setdefault("field2d_profile", {})[f"{e} {tag}"] = {k: (v["res"] if isinstance(v, dict) and "res" in v else v) for k, v in r.items()}
TAB = (r"""\begin{table}[H]
\centering
\caption{Two-dimensional potentials at $c=0.04$: relative profile errors (\%) of the density maps predicted by the functionals trained on planar profiles. As for the planar profiles, the error is $\|n^{\rm pred}_\alpha-n^{\rm MD}_\alpha\|_2/\|n^{\rm MD}_\alpha\|_2$, averaged over cations and anions. It is evaluated on the Fourier modes of the maps that MD resolves, those with amplitudes above three standard errors (their number for both species is given), because single bins of the MD maps carry a sampling noise of 5--6\%. The last column gives the MD noise level of the same quantity; an error at this level means agreement within the MD precision.}
\label{tab:S:field2d}
\begin{tabular}{llccccc}
$\varepsilon_r$ & potential & resolved modes & neural functional & pair closure & PB & MD noise\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}""")
path = f"{LX}/si/si_s4_validation.tex"; tex = open(path).read()
m = re.search(r"\\begin\{table\}\[H\]((?!\\begin\{table\}).)*?\\label\{tab:S:field2d\}.*?\\end\{table\}", tex, re.S); assert m, "field2d table"
tex = tex[:m.start()] + TAB + tex[m.end():]; open(path, "w").write(tex)
print("== field2d (profile errors)"); print("\n".join(rows))
json.dump(OUT, open(f"{FG}/single_model_metrics.json", "w"), indent=1, default=float)
print("== selection"); print(json.dumps(SEL, indent=0)[:3000])
