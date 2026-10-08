from figstyle import *

KK = json.load(open(os.path.join(ROOT, "results/figures/kernel_k1.json")))
LB = {"7.5": "7.96", "2": "29.8"}
W, H = 88, 82
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])
rep = {}
for lab, e, y0 in (("a", "7.5", 43), ("b", "2", 9)):
    d = KK[e]; SYS = system(e); gmd = json.load(open(SYS["gamma_md"]))
    nb = np.array([d["fun"][f"{c:g}"]["nbar"] for c in CONCS])
    g = np.array([gmd[f"{c:g}"]["shell1"]["Gamma"] for c in CONCS]); ge = np.array([gmd[f"{c:g}"]["shell1"]["Gamma_err"] for c in CONCS])
    sw = np.array([d["fun"][f"{c:g}"]["sum"] for c in CONCS]); swp = np.array([d["pair"][f"{c:g}"]["sum"] for c in CONCS])
    ax = ax_mm(17, y0, 67, 30)
    ax.plot(CONCS, sw, "-", marker="o", ms=2.4, color=C_FUN, lw=0.9)
    ax.plot(CONCS, swp, "--", marker="s", ms=2.4, color=C_PAIR, lw=0.9)
    ax.errorbar(CONCS, 2 * (g - 1) / nb, yerr=2 * ge / nb, fmt="o", ms=2.6, color=C_MD, capsize=1.5, lw=0.6, zorder=5)
    ax.axhline(0, color=BAND, lw=0.6); ax.set_xscale("log"); ax.set_xticks([0.01, 0.02, 0.04, 0.08]); ax.set_xticklabels(["0.01", "0.02", "0.04", "0.08"]); ax.minorticks_off()
    ax.set_ylabel(r"$\mathbf{q}^{\mathsf{T}}W(k_1)\,\mathbf{q}$ / $k_BT\sigma^3$", labelpad=1)
    ax.set_title(f"$\\varepsilon_r$ = {e}, $l_B$ = {LB[e]} $\\sigma$", loc="left", pad=2)
    panel(ax, lab, x=-0.24, y=1.04)
    if e == "7.5":
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("$c$", labelpad=1)
    rep[e] = {"MD_sum": dict(zip([f"{c:g}" for c in CONCS], (2 * (g - 1) / nb).round(1).tolist())),
              "MD_sum_err": dict(zip([f"{c:g}" for c in CONCS], (2 * ge / nb).round(1).tolist())),
              "fun_sum": dict(zip([f"{c:g}" for c in CONCS], sw.round(1).tolist())), "pair_sum": dict(zip([f"{c:g}" for c in CONCS], swp.round(1).tolist()))}

h = [plt.Line2D([], [], color=C_MD, marker="o", ms=2.6, lw=0), plt.Line2D([], [], color=C_FUN, marker="o", ms=2.6, lw=0.9),
     plt.Line2D([], [], color=C_PAIR, ls="--", marker="s", ms=2.6, lw=0.9)]
fig.legend(h, ["MD estimate", "neural functional", "pair closure"], loc="upper left", bbox_to_anchor=(17 / W, 1.0), ncol=3,
           handlelength=1.6, columnspacing=1.2, borderaxespad=0.3)
save(fig, "fig4")
json.dump(rep, open(os.path.join(ROOT, "results/figures/fig4_metrics.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
