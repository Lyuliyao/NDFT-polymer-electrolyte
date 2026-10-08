"""Table 1 of the main text and SI tab:S:archfrac in profile errors (2026-10-07; user: a chi^2 without a reference does not tell
the reader what is good, report profile errors).  Every entry is the mean relative L2 error (%) of the profiles predicted by the
Euler-Lagrange equation, cations and anions averaged, as in Fig. 2, for the reported model of each form (lowest validation
residual, make_tab_single.py).  Means run over the solutions that converge to a profile (Euler-Lagrange converged and error
below 50%); the number of the other test solutions (not converged, or collapsed onto a density spike) is counted for the
footnotes.  The convolutional free energy converges for almost no run: its entries are the errors of the final iterates.
Reference: the statistical error of the MD profiles, ||s_n|| / ||n||, averaged over the 42 test runs of each eps_r.
    NDFT_LATEX=<tree> python make_tab_profile.py   -> Table 1 body of main.tex, tab:S:archfrac of si/si_s4_validation.tex,
                                                     paper/figures/profile_table_metrics.json"""
import json, os, re, subprocess, sys
import numpy as np
R = "/mnt/research/MultiscaleML_group/Liyao"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"; FG = f"{S}/paper/figures"
LX = os.environ.get("NDFT_LATEX", f"{S}/paper/output/latex"); T = "c001_c002_c004_c006_c008"
ROOT = {"7.5": f"{R}/salt_in_polymer_eps75/runs/learn", "2": f"{R}/salt_in_polymer_eps2/runs/learn"}
SEL = json.load(open(f"{FG}/single_model_metrics.json"))["selection"]
PAIR = {"7.5": f"joint_v1_L1_C4_R128x128_H64_{T}_val3_kn1softplus_kbK2_long_s2", "2": f"joint_v1_L1_C4_R128x128_H64_{T}_val3_kn1softplus_kbK2_s0"}   # 2026-10-08: pair closure with the learned kernels of the functional (seeds by validation)
COLLAPSE = 50.0
isp27 = lambda e, r: e == "2" and r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9
l2 = lambda r: 100 * float(np.mean([r["profile"][s]["rel_l2"] for s in ("cation", "anion") if s in r["profile"]]))
ok = lambda r: r.get("el", {}).get("converged", True) and l2(r) <= COLLAPSE
fmt = lambda x: "--" if x is None or not np.isfinite(x) else (f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}"))
OUT = {}


def stats(root, model, e, final_iterates=False):
    """held-out (eps 2: without p27), c = 0.05, p27; failures among the test solutions"""
    m = json.load(open(f"{root}/{model}/metrics.json")); p5f = f"{root}/{model}/predict_c05/metrics.json" if not os.path.exists(f"{root}/{model}/predict_c005/metrics.json") else f"{root}/{model}/predict_c005/metrics.json"
    p = json.load(open(p5f)) if os.path.exists(p5f) else {"transfer_rows": []}
    ho = m["heldout_rows"]; t5 = p["transfer_rows"]; q = []          # 2026-10-07 (user): all held-out runs, no special case
    use = (lambda rs: rs) if final_iterates else (lambda rs: [r for r in rs if ok(r)])
    mean = lambda rs: float(np.mean([l2(r) for r in use(rs)])) if use(rs) else float("nan")
    tests = m["heldout_rows"] + t5
    return dict(ho=mean(ho), c5=mean(t5), p27=(l2(q[0]) if q and (final_iterates or ok(q[0])) else (float("nan") if q else None)),
                notconv=sum(not r.get("el", {}).get("converged", True) for r in tests),
                collapsed=sum(r.get("el", {}).get("converged", True) and l2(r) > COLLAPSE for r in tests), ntest=len(tests),
                ho_bad=sum(not ok(r) for r in ho))


rows = []; foot = {}
LJ = f"{R}/salt_in_polymer_lj1/runs/learn"; LJT = "c01_c02_c04_c06_c07"
LJN = {"ours": "joint_v2_L1_C4_R128x128_H64_{T}_val3_kn1softplus{f}_kbK2_lj1_s{s}", "c1win": "joint_c1win_W30_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj1_s{s}",
       "cace": "joint_cace_a5q3_L1_C4_R128x128_H64_{T}_val3{f}_kbK2_lj1_s{s}"}
LJK = {"ours": "Neural functional (18{,}861)", "c1win": "One-body network, $\\pm3\\sigma$ (24{,}577)", "cace": "Lattice free energy, $R=1.5\\sigma$ (19{,}426)"}


def ljstats(form, frac):
    key = f"LJ {LJK[form]} {'full' if frac == '' else 'f025'}"
    if key in SEL:
        s = SEL[key]["seed"]
    else:                                   # the quarter selection was not stored by make_tab_single: lowest validation residual
        vs = [json.load(open(f"{LJ}/{LJN[form].format(T=LJT, f=frac, s=k)}/metrics.json")).get("val_chi2", np.inf) for k in range(3)]; s = int(np.argmin(vs))
    return stats(LJ, LJN[form].format(T=LJT, f=frac, s=s), "lj")


for form, lab, npar in (("ours", "Neural functional (this work)", "19{,}641"), ("c1win", "One-body network, $\\pm3\\sigma$$^{\\S}$", "32{,}514"), ("cace", "Lattice free energy$^{\\dagger}$", "23{,}828")):
    a = {e: stats(ROOT[e], SEL[f"{form} {e}"]["model"], e) for e in ("7.5", "2")}
    q = {e: stats(ROOT[e], SEL[f"{form}_f025 {e}"]["model"], e) for e in ("7.5", "2")}
    lf, lq = ljstats(form, ""), ljstats(form, "_f025")
    struct = re.search(r"Neural functional \(this work\).*", "")  # placeholder
    rows.append([lab, npar, fmt(lf["ho"]), fmt(lq["ho"]), fmt(a["7.5"]["ho"]), fmt(a["2"]["ho"]), fmt(a["7.5"]["c5"]), fmt(a["2"]["c5"]), fmt(q["7.5"]["ho"]), fmt(q["2"]["ho"])])
    OUT[form] = dict(full=a, quarter=q, lj=lf, lj_quarter=lq)
cnn = {e: stats(ROOT[e], SEL[f"cnn {e}"]["model"], e, final_iterates=True) for e in ("7.5", "2")}
# Lennard-Jones fluid (2026-10-07, user: why was the convolutional free energy not run on the LJ fluid?): lj1_learn_cnn.sbatch
LJN["cnn"] = "joint_cnn_d2_L1_C4_{T}_val3{f}_kbK2_lj1_s{s}"
def ljcnn(frac):
    """the CNN on the LJ fluid: most of its Euler-Lagrange solutions do not converge either, so, as on the ions, its entries are
    the errors of the final iterates (the selected seed needs its predictions; the others only their validation residual)"""
    names = [LJN["cnn"].format(T=LJT, f=frac, s=k) for k in range(3)]
    if not all(os.path.exists(f"{LJ}/{n}/metrics.json") for n in names):
        return None
    vs = [json.load(open(f"{LJ}/{n}/metrics.json")).get("val_chi2", np.inf) for n in names]; k = int(np.argmin(vs))
    if not os.path.exists(f"{LJ}/{names[k]}/predict_c05/metrics.json"):
        return None
    st = stats(LJ, names[k], "lj", final_iterates=True); st_ok = stats(LJ, names[k], "lj")
    st.update(seed=k, val=vs, ho_converged_only=st_ok["ho"]); return st
cl, cq = ljcnn(""), ljcnn("_f025")
OUT["cnn_lj"] = dict(full=cl, quarter=cq)
rows.append(["Convolutional free energy$^{\\ddagger}$", "24{,}321", fmt(cl["ho"]) if cl else "--", fmt(cq["ho"]) if cq else "--", fmt(cnn["7.5"]["ho"]), fmt(cnn["2"]["ho"]), fmt(cnn["7.5"]["c5"]), fmt(cnn["2"]["c5"]), "--", "--"])
OUT["cnn"] = cnn
pa = {e: stats(ROOT[e], PAIR[e], e) for e in ("7.5", "2")}; pl = stats(LJ, f"joint_v1_L1_C4_R128x128_H64_{LJT}_val3_kn1softplus_kbK2_lj1_s2", "lj")
rows.append(["Pair closure", "1{,}336", fmt(pl["ho"]), "--", fmt(pa["7.5"]["ho"]), fmt(pa["2"]["ho"]), fmt(pa["7.5"]["c5"]), fmt(pa["2"]["c5"]), "--", "--"])
OUT["pair"] = dict(full=pa, lj=pl)

# MD noise of the test profiles (reference)
noise = {}
for e, sysn in (("7.5", "eps75"), ("2", "eps2")):
    code = f"""
import json, numpy as np, learn.data as D
sps, runs = D.load_all(); ho = D.heldout_tags(runs)
test = [r for r in runs if r.status == "PASS" and ((r.conc in ho and r.tag in ho[r.conc]) or (abs(r.conc - 0.05) < 1e-9 and int(r.tag[1:]) <= 21))]
v = [100 * 0.5 * sum(np.linalg.norm(r.sig_n[a]) / np.linalg.norm(r.n[a]) for a in range(2)) for r in test]
print(json.dumps(dict(mean=float(np.mean(v)), n=len(v))))"""
    env = dict(os.environ, SIP_ROOT=f"{R}/salt_in_polymer_{sysn}", JAX_PLATFORMS="cpu", PYTHONNOUSERSITE="1")
    res = subprocess.run([sys.executable, "-c", code], cwd=f"{R}/salt_in_polymer_eps2/code", env=env, capture_output=True, text=True)
    noise[e] = json.loads(res.stdout.strip().splitlines()[-1])
OUT["md_noise"] = noise

# structure column: keep the one written by make_tab_single.py
tex = open(f"{LX}/main.tex").read()
m = re.search(r"(\\label\{tab:compare\}.*?)(\\begin\{tabular\}\{[^}]*\}\n)(.*?)(\\bottomrule)", tex, re.S); assert m
old_rows = [ln for ln in m.group(3).split("\n") if ln.strip().endswith("\\\\") and "&" in ln and not ln.lstrip().startswith("Form") and "multicolumn" not in ln]
struct = {ln.split("&")[0].strip().replace("$^{\\S}$", ""): ln.rsplit("&", 1)[1].strip()[:-2].strip() for ln in old_rows}
header = r""" & & \multicolumn{2}{c}{Lennard-Jones fluid} & \multicolumn{2}{c}{ions, held-out} & \multicolumn{2}{c}{$c=0.05$} & \multicolumn{2}{c}{quarter of runs} & structure$^{*}$\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Form & parameters & all runs & quarter & $\varepsilon_r=7.5$ & 2 & 7.5 & 2 & 7.5 & 2 & force; Hessian; path\\
\midrule
"""
body = "\n".join(" & ".join(r) + " & " + struct[r[0].replace("$^{\\S}$", "")] + "\\\\" for r in rows)
tex = tex[:m.start(2)] + "\\begin{tabular}{lcccccccccc}\n" + header + body + "\n" + tex[m.start(4):]
open(f"{LX}/main.tex", "w").write(tex)
print("== Table 1 (profile errors)"); print(body)
print("MD noise of the test profiles:", {e: round(v["mean"], 2) for e, v in noise.items()})
fl = lambda d: f"not converged {d['notconv']}, collapsed {d['collapsed']} of {d['ntest']}"
for f in ("ours", "c1win", "cace"):
    print(f, "| ions full:", {e: fl(OUT[f]["full"][e]) for e in ("7.5", "2")}, "| quarter held-out bad:", {e: OUT[f]["quarter"][e]["ho_bad"] for e in ("7.5", "2")}, "| LJ:", fl(OUT[f]["lj"]), "quarter", fl(OUT[f]["lj_quarter"]))
print("cnn", {e: fl(cnn[e]) for e in ("7.5", "2")}, "| pair", {e: fl(pa[e]) for e in ("7.5", "2")}, "LJ", fl(pl))
for lab, st in (("cnn LJ full", cl), ("cnn LJ quarter", cq)):
    if st: print(lab, "seed", st["seed"], "val", [round(v, 2) for v in st["val"]], "held-out (final iterates) %.2f, converged only %.2f" % (st["ho"], st["ho_converged_only"]), fl(st))

# SI tab:S:archfrac: held-out / c = 0.05 profile errors
frows = []
for e in ("7.5", "2"):
    for frac, suf in (("0.25", "_f025"), ("0.5", "_f050"), ("1", "")):
        cells = []
        for form in ("ours", "c1win", "cace"):
            st = stats(ROOT[e], SEL[f"{form}{suf} {e}"]["model"], e); cells.append(f"{fmt(st['ho'])} / {fmt(st['c5'])}")
            OUT.setdefault("archfrac", {})[f"{form} {e} {frac}"] = st
        frows.append(f"{e} & {frac} & " + " & ".join(cells) + "\\\\")
path = f"{LX}/si/si_s4_validation.tex"; tex = open(path).read()
m = re.search(r"(\\label\{tab:S:archfrac\}.*?\\midrule\n)(.*?)(\\bottomrule)", tex, re.S); assert m
tex = tex[:m.start(2)] + "\n".join(frows) + "\n" + tex[m.start(3):]; open(path, "w").write(tex)
print("== archfrac (profile errors)"); print("\n".join(frows))
json.dump(OUT, open(f"{FG}/profile_table_metrics.json", "w"), indent=1, default=float)
