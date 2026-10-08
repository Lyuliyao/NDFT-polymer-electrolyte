"""SI Appendix, section S10: every held-out profile at one eps_r (--eps 7.5 | 2).
Rows: the five training concentrations; columns: the four held-out runs of each (one per potential stratum).
Each panel: cation (upper curves) and anion (lower curves, divided by 3 for legibility) densities n_alpha / nbar on a log axis,
MD (black, +-2 sigma band) against the pair closure (orange dashed) and the neural functional (blue), both solved from V(z) alone.
Numbers in each panel: relative L2 error, cation / anion, neural functional (blue) and pair closure (orange)."""
from figstyle import *
from learn import data as D

EPS = eps_arg(); SYS = system(EPS)
TRAIN_C = [0.01, 0.02, 0.04, 0.06, 0.08]
sps, runs = D.load_all(root=SYS["root"]); ho = D.heldout_tags(runs, root=SYS["root"])
pred = {m: np.load(os.path.join(d, "heldout.npz")) for m, d in (("pair", SYS["pair"]), ("fun", SYS["fun"][0]))}
ANI_SHIFT = 1 / 3

def rl2(p, n): return float(np.sqrt(np.mean((p - n) ** 2) / np.mean(n ** 2)))

W, H = 178, 200
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])
out = {}
for i, c in enumerate(TRAIN_C):
    tags = sorted(ho[c]); geo = D.geometry_of(sps[c], int(round(sps[c].L / 0.1))); nb = geo.nbar
    for j, tag in enumerate(tags):
        r = [x for x in runs if x.conc == c and x.tag == tag][0]; g = D.geometry_of(sps[c], r.n.shape[1]); z = (np.arange(r.n.shape[1]) + 0.5) * g.dz
        ax = ax_mm(14 + j * 41, H - 17 - 28 - i * 37, 35, 28)
        key = f"c{c:g}_{tag}"; err = {}
        for a_, sh in ((0, 1.0), (1, ANI_SHIFT)):
            ax.fill_between(z, sh * (r.n[a_] - 2 * r.sig_n[a_]) / nb, sh * (r.n[a_] + 2 * r.sig_n[a_]) / nb, color=BAND, lw=0)
            ax.plot(z, sh * r.n[a_] / nb, color=C_MD, lw=0.9)
            for m, col, ls in (("pair", C_PAIR, "--"), ("fun", C_FUN, "-")):
                p = pred[m][key][a_]; err[(m, a_)] = rl2(p, r.n[a_])
                ax.plot(z, sh * p / nb, ls, color=col, lw=0.75)
        ax.set_yscale("log"); ax.set_xlim(0, g.L); ax.minorticks_off()
        ax.text(0.0, 1.03, f"{tag}", transform=ax.transAxes, va="bottom", ha="left", fontsize=5.8, color=INK2, weight="bold")
        ax.text(1.0, 1.13, f"{100 * err[('fun', 0)]:.1f} / {100 * err[('fun', 1)]:.1f}%", transform=ax.transAxes, va="bottom", ha="right", fontsize=5.2, color=C_FUN)
        ax.text(1.0, 1.02, f"{100 * err[('pair', 0)]:.1f} / {100 * err[('pair', 1)]:.1f}%", transform=ax.transAxes, va="bottom", ha="right", fontsize=5.2, color=C_PAIR)
        if i == len(TRAIN_C) - 1: ax.set_xlabel(r"$z/\sigma$", labelpad=1)
        else: ax.set_xticklabels([])
        if j == 0: ax.set_ylabel(f"$c$ = {c:g}\n" + r"$n_\alpha/\bar n$", labelpad=1)
        out[f"c{c:g}/{tag}"] = {m: [round(err[(m, 0)], 4), round(err[(m, 1)], 4)] for m in ("fun", "pair")}
h = [plt.Line2D([], [], color=C_MD, lw=1.0), plt.Line2D([], [], color=C_PAIR, ls="--", lw=0.9), plt.Line2D([], [], color=C_FUN, lw=0.9)]
fig.legend(h, ["MD ($\\pm$2 s.e. band)", "pair closure", "neural functional (seed 0)"], loc="upper left", bbox_to_anchor=(14 / W, 1.0), ncol=3, handlelength=1.8, columnspacing=1.6)
fig.text(1 - 4 / W, 1 - 3 / H, f"$\\varepsilon_r$ = {EPS}, $l_B$ = {SYS['lB']} $\\sigma$", ha="right", va="top", fontsize=6.5)
save(fig, f"si/heldout{SYS['suffix']}")
json.dump(out, open(os.path.join(ROOT, f"paper/figures/si/heldout{SYS['suffix']}_metrics.json"), "w"), indent=1)
print(json.dumps(out, indent=0))
