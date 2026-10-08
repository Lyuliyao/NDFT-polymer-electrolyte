"""SI tables of section S4.4 regenerated from the models: tab:S:res75, tab:S:res2 (held-out residuals and profile errors by
concentration) and tab:S:sk (structure-factor errors and Gamma(k_1)).  Prints the table bodies and splices them into
si_s4.tex between \\midrule and \\bottomrule of each table (captions and headers unchanged).
    python make_tab_results.py          (2026-10-04, spectrum-loss models via figstyle.system)"""
from figstyle import *
from learn import data as D
from learn.models import W_of_k, fine_geometry
from learn.evaluate import S_from_W
from learn.protocol import load_model
import re
L = os.path.join(ROOT, "paper/output/latex/si/si_s4.tex")
F2 = json.load(open(os.path.join(ROOT, "paper/figures/fig2_metrics.json")))
CS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]
fmt3 = lambda v: f"{v:.3f}"; pm = lambda v: f"${np.mean(v):.3f}\\pm{np.std(v, ddof=1):.3f}$"
out_tables = {}
gam = {}
for e in ("7.5", "2"):
    SYS = system(e); sps, runs = D.load_all(root=SYS["root"])
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CS}
    mets = [json.load(open(os.path.join(d, "metrics.json"))) for d in SYS["fun"]]
    p5 = [json.load(open(os.path.join(d, "predict_c005/metrics.json"))) for d in SYS["fun"]]
    pair = json.load(open(os.path.join(SYS["pair"], "metrics.json"))); pair5 = json.load(open(os.path.join(SYS["pair"], "predict_c005/metrics.json")))
    isp27 = lambda r: e == "2" and r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9
    def sel(rows, c, p27=None):
        return [r for r in rows if abs(r["conc"] - c) < 1e-9 and (p27 is None or isp27(r) == p27)]
    chi = lambda rows: float(np.mean([r["chi2"] for r in rows])) if rows else float("nan")
    prof = lambda rows: (100 * np.mean([r["profile"]["cation"]["rel_l2"] for r in rows]), 100 * np.mean([r["profile"]["anion"]["rel_l2"] for r in rows]))
    pb_chi = lambda keys: float(np.mean([F2["chi2_7.5"]["PB"][k] for k in keys])) if e == "7.5" else None
    pb_prof = lambda keys: 100 * float(np.mean([F2[e]["PB"][k] for k in keys]))
    rows = []; ptrain = []
    def line(label, nf_rows, nf5, pr_rows, pr5, keys, train_c=None):
        if nf5 is None:
            c0 = chi(nf_rows[0]); c3 = [chi(r) for r in nf_rows]; pc, pa = prof(nf_rows[0]); tr = fmt3(train_c) if train_c is not None else "--"
            pchi = chi(pr_rows); ppc, ppa = prof(pr_rows)
        else:
            c0 = nf5[0]["transfer"]["chi2"]; c3 = [x["transfer"]["chi2"] for x in nf5]; pc, pa = 100 * nf5[0]["transfer"]["cation_rel_l2"], 100 * nf5[0]["transfer"]["anion_rel_l2"]; tr = "--"
            pchi = pr5["transfer"]["chi2"]; ppc, ppa = 100 * pr5["transfer"]["cation_rel_l2"], 100 * pr5["transfer"]["anion_rel_l2"]
        seeds = (", ".join(f"{v:.2f}" for v in c3) + "$^{*}$") if label == "0.04, p27" else pm(c3)     # p27: the three seeds listed, as before
        s = f"{label} & {fmt3(c0)} & {seeds} & {pc:.1f} / {pa:.1f} & {tr} & {pchi:.2f} & {ppc:.1f} / {ppa:.1f}"
        if e == "7.5":
            pbc = pb_chi(keys); s += f" & {pbc:.0f} & {pb_prof(keys):.1f}" if pbc >= 10 else f" & {pbc:.1f} & {pb_prof(keys):.1f}"
        rows.append(s + r"\\")
    keys_of = lambda rs: [f"c{r['conc']:g}/{r['tag']}" for r in rs]
    for c in CS:
        if c == 0.05:
            keys = [k for k in F2[e]["PB"] if k.startswith("c0.05/")]
            line("0.05", None, p5, None, pair5, keys); continue
        trc = np.mean([r["chi2"] for r in mets[0]["train_rows"] if abs(r["conc"] - c) < 1e-9])
        ptrain.append(np.mean([r["chi2"] for r in pair["train_rows"] if abs(r["conc"] - c) < 1e-9]))
        if e == "2" and c == 0.04:
            line("0.04 (without p27)", [sel(m["heldout_rows"], c, False) for m in mets], None, sel(pair["heldout_rows"], c, False), None, keys_of(sel(mets[0]["heldout_rows"], c, False)), trc)
            line("0.04, p27", [sel(m["heldout_rows"], c, True) for m in mets], None, sel(pair["heldout_rows"], c, True), None, keys_of(sel(mets[0]["heldout_rows"], c, True)), None)
        else:
            line(f"{c:g}", [sel(m["heldout_rows"], c) for m in mets], None, sel(pair["heldout_rows"], c), None, keys_of(sel(mets[0]["heldout_rows"], c)), trc)
    allho = [m["heldout_rows"] for m in mets]
    line("20 held-out", allho, None, pair["heldout_rows"], None, keys_of(mets[0]["heldout_rows"]), np.mean([r["chi2"] for r in mets[0]["train_rows"]]))
    if e == "2":
        line("19 held-out (without p27)", [[r for r in m["heldout_rows"] if not isp27(r)] for m in mets], None, [r for r in pair["heldout_rows"] if not isp27(r)], None, keys_of([r for r in mets[0]["heldout_rows"] if not isp27(r)]), None)
    note = "Training $\\chi^2$ of the pair closure: " + ", ".join(f"{v:.2f}" for v in ptrain) + f" at $c=0.01$, 0.02, 0.04, 0.06, 0.08 ({np.mean([r['chi2'] for r in pair['train_rows']]):.2f} overall)."
    out_tables["tab:S:res75" if e == "7.5" else "tab:S:res2"] = (rows, note)
    # ---- Gamma(k1) of the functional (3 seeds) and the pair closure
    gam[e] = {}
    for lab, dirs in (("fun", SYS["fun"]), ("pair", [SYS["pair"]])):
        vals = {c: [] for c in CS}
        for d in dirs:
            cfg, ps = load_model(d)
            for c in CS:
                g = geoms[c]; gf = fine_geometry(g, 8); k = np.asarray(gf.k)
                SNN = S_from_W(np.asarray(W_of_k(ps, cfg, gf)), k, g.nbar, g.lB)["NN"] / (2 * g.nbar)
                vals[c].append(float(1 / SNN[int(np.argmin(abs(k - 2 * np.pi / g.L)))]))
        gam[e][lab] = vals
# ---- tab:S:sk from structure_metrics (7.5, all six c) and si/structure_eps2_metrics (2), Gamma MD and tau_int kept from the file
sm = {"7.5": json.load(open(os.path.join(ROOT, "paper/figures/structure_metrics.json"))), "2": json.load(open(os.path.join(ROOT, "paper/figures/si/structure_eps2_metrics.json")))}
tex = open(L).read()
old = re.search(r"\\label\{tab:S:sk\}.*?\\midrule(.*?)\\bottomrule", tex, re.S).group(1)
md_cols = {}
for ln in old.strip().splitlines():
    if ln.startswith("\\multicolumn"): blk = "7.5" if "7.5" in ln else "2"; continue
    if ln.strip() in ("\\midrule", ""): continue
    cells = [x.strip() for x in ln.rstrip("\\").split("&")]
    md_cols[(blk, cells[0])] = (cells[5], cells[6])
sk_rows = []
for e in ("7.5", "2"):
    sk_rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{$\\varepsilon_r={e}$}}}}\\\\")
    for c in CS:
        cs = f"{c:g}"; s = sm[e]
        zz = [100 * s[f"fun_s{i}"][cs]["ZZ"] for i in range(3)]; nn = [100 * s[f"fun_s{i}"][cs]["NN"] for i in range(3)]
        gmd, tau = md_cols[(e, cs)]
        gf = gam[e]["fun"][c]; gp = gam[e]["pair"][c][0]
        sk_rows.append(f"{cs} & {zz[0]:.1f} (${np.mean(zz):.1f}\\pm{np.std(zz, ddof=1):.1f}$) & {100 * s['pair'][cs]['ZZ']:.1f} & {nn[0]:.1f} (${np.mean(nn):.1f}\\pm{np.std(nn, ddof=1):.1f}$) & {100 * s['pair'][cs]['NN']:.1f} & {gmd} & {tau} & {gf[0]:.2f} (${np.mean(gf):.2f}\\pm{np.std(gf, ddof=1):.2f}$) & {gp:.2f}\\\\")
    if e == "7.5": sk_rows.append("\\midrule")
out_tables["tab:S:sk"] = (sk_rows, None)
# ---- splice
for lab, (rows, note) in out_tables.items():
    m = re.search(r"(\\label\{" + re.escape(lab) + r"\}.*?\\midrule\n)(.*?)(\\bottomrule)", tex, re.S); assert m, lab
    tex = tex[:m.start(2)] + "\n".join(rows) + "\n" + tex[m.start(3):]
    if note:
        m2 = re.search(r"(\\label\{" + re.escape(lab) + r"\}.*?\\addtabletext\{)(.*?)(Training \$\\chi\^2\$ of the pair closure: [^}]*\})", tex, re.S); assert m2, lab + " note"
        tex = tex[:m2.start(3)] + note + "}" + tex[m2.end(3):]
    print("==", lab); print("\n".join(rows))
open(L, "w").write(tex); print("si_s4.tex tables spliced")
json.dump({e: {lab: {f"{c:g}": v for c, v in d.items()} for lab, d in gam[e].items()} for e in gam}, open(os.path.join(ROOT, "paper/figures/si/gamma_k1_models.json"), "w"), indent=1)
