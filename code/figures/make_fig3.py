from figstyle import *
import matplotlib.patches
from learn import data as D
from learn.train import residual_pulls
from learn.protocol import load_model
from learn.models import W_of_k, fine_geometry
from learn.evaluate import S_from_W, bulk_structure

M = {e: {"pair": [system(e)["pair"]], "fun": system(e)["fun"]} for e in ("7.5", "2")}
ROOTS = {e: system(e)["root"] for e in ("7.5", "2")}
OUT = "p27"


def gamma_k1(d, root):

    cf, ps = load_model(d); sps_, _ = D.load_all(root=root, concs=CONCS); out = {}
    for c in CONCS:
        g = D.geometry_of(sps_[c], int(round(sps_[c].L / 0.1))); gf = fine_geometry(g, 8); k = np.asarray(gf.k)
        S = S_from_W(np.asarray(W_of_k(ps, cf, gf)), k, g.nbar, g.lB)["NN"] / (2 * g.nbar)
        out[f"{c:g}"] = float(1 / S[int(np.argmin(abs(k - 2 * np.pi / g.L)))])
    return out


gam = {e: {"pair": gamma_k1(M[e]["pair"][0], ROOTS[e]), "fun": [gamma_k1(d, ROOTS[e]) for d in M[e]["fun"]]} for e in ROOTS}
gmd = {e: json.load(open(system(e)["gamma_md"])) for e in ROOTS}


chi = {e: {m: [] for m in ("pair", "fun")} for e in ROOTS}
p27 = {m: [] for m in ("pair", "fun")}
for e, rt in ROOTS.items():
    sps, runs = D.load_all(root=rt, uncertainty="legacy"); ho = D.heldout_tags(runs, root=rt); test = test_runs(runs, ho)
    for m in ("pair", "fun"):
        for d in M[e][m]:
            cfg, ps = load_model(d); per = {}
            for c in CONCS:
                rs = [r for r in test if r.conc == c]; b, _ = D.make_batches(rs, sps)[c]
                v = (np.asarray(residual_pulls(ps, cfg, b)) ** 2).reshape(len(rs), 2, -1).mean(axis=2).mean(axis=1)
                keep = np.array([not (e == "2" and c == 0.04 and r.tag == OUT) for r in rs])
                per[c] = float(v[keep].mean())
                if not keep.all(): p27[m].append(float(v[~keep][0]))
            chi[e][m].append(per)


W_, H = 89, 86
fig = plt.figure(figsize=(W_ * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / W_, y / H, w / W_, h / H])
axz, axn = ax_mm(10, 50, 33, 29), ax_mm(54, 50, 33, 29)
axg = {"7.5": ax_mm(10, 8, 33, 29), "2": ax_mm(54, 8, 33, 29)}

S75 = system("7.5"); sps75, _ = D.load_all(root=S75["root"])
cfg, ps = load_model(S75["fun"][0])
PLOT_C = [c for c in CONCS if c != 0.01]
cols = cramp(PLOT_C)
for c in PLOT_C:
    g = D.geometry_of(sps75[c], int(round(sps75[c].L / 0.1)))
    sk = sps75[c].sk; n2 = 2 * float(sk["n_each"]); st = bulk_structure(ps, cfg, g); kk = st["S"]["k"]; sel = (kk > 0) & (kk <= 2.1)
    for ax, comp in ((axz, "ZZ"), (axn, "NN")):
        ax.plot(sk["k"], sk[f"S_{comp}"] / n2, "o", ms=1.3, color=cols[c], lw=0, zorder=3)
        ax.plot(kk[sel], st["S"][comp][sel] / n2, "-", color=cols[c], lw=0.8, zorder=2, label=f"{c:g}" + (" (untrained)" if c == 0.05 else ""))
for ax, comp, lab, top in ((axz, "ZZ", "charge", 2.05), (axn, "NN", "number", 1.0)):
    ax.set_xlim(0, 2.15); ax.set_ylim(0, top); ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel(r"$k\sigma$", labelpad=1); ax.set_ylabel(rf"$S_{{{comp}}}(k)\,/\,2\bar n$", labelpad=1)
    ax.set_title(f"{lab}, $\\varepsilon_r$ = 7.5", loc="left", pad=2)
axz.legend(title="$c$", loc="upper left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.25, title_fontsize=5.5)
axn.text(0.03, 0.97, "points: MD\nlines: neural functional", transform=axn.transAxes, fontsize=5.2, color=INK2, va="top", ha="left")
panel(axz, "a", x=-0.28, y=1.04); panel(axn, "b", x=-0.28, y=1.04)

for e, ax in axg.items():
    gf_ = np.array([[s[f"{c:g}"] for c in CONCS] for s in gam[e]["fun"]])
    ax.axhline(1, color="#b0afa9", lw=0.6, ls=":")
    ax.plot(CONCS, gf_[0], "-", marker="o", ms=2.0, color=C_FUN, lw=0.9)
    ax.plot(CONCS, [gam[e]["pair"][f"{c:g}"] for c in CONCS], "--", marker="s", ms=1.9, color=C_PAIR, lw=0.8)
    ax.errorbar(CONCS, [gmd[e][f"{c:g}"]["shell1"]["Gamma"] for c in CONCS], yerr=[gmd[e][f"{c:g}"]["shell1"]["Gamma_err"] for c in CONCS],
                fmt="o", ms=2.2, color=C_MD, capsize=1.2, lw=0.6, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xticks([0.01, 0.02, 0.04, 0.08]); ax.set_xticklabels(["0.01", "0.02", "0.04", "0.08"])
    ax.set_yticks([0.3, 1, 3, 10]); ax.set_yticklabels(["0.3", "1", "3", "10"]); ax.set_ylim(0.25, 16)
    ax.set_xlabel("$c$", labelpad=1); ax.set_ylabel(r"$\Gamma = 2\bar n/S_{NN}(k_1)$", labelpad=1)
    ax.set_title(f"$\\varepsilon_r$ = {e}, $l_B$ = {'7.96' if e == '7.5' else '29.8'}$\\sigma$", loc="left", pad=2)
h = [plt.Line2D([], [], color=C_MD, marker="o", ms=2.2, lw=0), plt.Line2D([], [], color=C_FUN, marker="o", ms=2.0, lw=0.9),
     plt.Line2D([], [], color=C_PAIR, ls="--", marker="s", ms=1.9, lw=0.8)]
axg["7.5"].legend(h, ["MD", "neural functional", "pair closure"], loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
panel(axg["7.5"], "c", x=-0.28, y=1.04); panel(axg["2"], "d", x=-0.28, y=1.04)
save(fig, "fig3")


fig = plt.figure(figsize=(W_ * MM, 44 * MM)); xs = np.arange(len(CONCS))
for i, e in enumerate(("7.5", "2")):
    ax = fig.add_axes([(10 + 44 * i) / W_, 10 / 44, 33 / W_, 27 / 44])
    for m, col, mk, ls in (("pair", C_PAIR, "s", "--"), ("fun", C_FUN, "o", "-")):
        arr = np.array([[s[c] for c in CONCS] for s in chi[e][m]])
        if len(arr) > 1: ax.vlines(xs, arr.min(0), arr.max(0), color=col, lw=2.0, alpha=0.35)
        ax.plot(xs, arr[0], ls, marker=mk, ms=2.2, color=col, lw=0.8, label={"pair": "pair closure", "fun": "neural functional"}[m])
    if e == "2":
        xx = CONCS.index(0.04) + 0.3
        for m, col, mk in (("pair", C_PAIR, "s"), ("fun", C_FUN, "o")):
            ax.plot([xx] * len(p27[m]), p27[m], mk, ms=1.8, color=col, mfc="white", mew=0.6)
        ax.annotate("p27", (xx, max(p27["pair"])), xytext=(3, 0), textcoords="offset points", fontsize=5, color=INK2, va="center")
    ax.axhline(1, color="#b0afa9", lw=0.6, ls="--"); ax.set_yscale("log"); ax.set_ylim(0.4, 150)
    ax.set_xticks(xs); ax.set_xticklabels([f"{c:g}" for c in CONCS], fontsize=4.8); ax.set_xlabel("$c$", labelpad=1)
    ax.set_ylabel(r"test $\chi^2$ per bin", labelpad=1); ax.set_title(f"$\\varepsilon_r$ = {e}", loc="left", pad=2)
    if i == 0: ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2)
    panel(ax, "ab"[i], x=-0.28, y=1.04)
save(fig, "si/chi2_conc")

rep = {"test_chi2": {e: {m: [{f"{c:g}": v for c, v in s.items()} for s in chi[e][m]] for m in chi[e]} for e in chi},
       "p27_eps2_chi2": p27,
       "Gamma": {e: {"MD": {f"{c:g}": [gmd[e][f"{c:g}"]["shell1"]["Gamma"], gmd[e][f"{c:g}"]["shell1"]["Gamma_err"]] for c in CONCS},
                     "pair": {f"{c:g}": gam[e]["pair"][f"{c:g}"] for c in CONCS},
                     "fun_seeds": [{f"{c:g}": s[f"{c:g}"] for c in CONCS} for s in gam[e]["fun"]]} for e in ROOTS}}
json.dump(rep, open(os.path.join(ROOT, "results/figures/fig3_metrics.json"), "w"), indent=1)
for e in ROOTS:
    gf_ = np.array([[s[f"{c:g}"] for c in CONCS] for s in gam[e]["fun"]])
    print(e, "Gamma reported functional", np.round(gf_[0], 2), "pair", [round(gam[e]["pair"][f"{c:g}"], 2) for c in CONCS])
