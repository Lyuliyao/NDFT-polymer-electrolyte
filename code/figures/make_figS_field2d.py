"""SI figure: two-dimensional test.  The functional trained on planar profiles predicts n_+(x, y) under a
potential V_alpha(x, y) (learn.field2d; nothing retrained), c = 0.04, both eps_r.  Per row: MD map, neural
functional (seed 0), their difference in units of the MD standard error, and a cut through the map
(MD band, neural functional, pair closure, Poisson-Boltzmann).  Data: <scratch>/field2d/<sys>/field_<tag>/
profiles2d.npz (tools/analyze_profiles2d.py) and the prediction cache of figs/field2d_compare.py.
    python make_figS_field2d.py [p52]"""
from figstyle import *
from matplotlib.colors import TwoSlopeNorm
B = "/mnt/research/MultiscaleML_group/Liyao"
TAG = sys.argv[1] if len(sys.argv) > 1 else "p52"
SYSN = (("eps75", "7.5", "7.96"), ("eps2", "2", "29.8"))
W_, H = 183, 100
fig = plt.figure(figsize=(W_ * MM, H * MM))
def ax_mm(x, y, w, h): return fig.add_axes([x / W_, y / H, w / W_, h / H])
out = {}
for r, (sysname, e, lb) in enumerate(SYSN):
    md = np.load(f"{ROOT}/field2d/{sysname}_c0.04/field_{TAG}/profiles2d.npz", allow_pickle=True)
    pr = np.load(f"{B}/salt_in_polymer_field2d/{sysname}/{TAG}_pred.npz", allow_pickle=True)
    assert np.allclose(pr["lo"], md["lo"]), "prediction cache does not use the MD box origin"
    L = float(md["lz"]); x = md["x"]; nm, er = md["cation_n"], md["cation_n_err"]; nb = nm.mean()
    nf, pc, pb = pr["n_NF s0"][0], pr["n_pair closure"][0], pr["n_PB"][0]
    y0 = int(np.unravel_index(np.argmax(nm), nm.shape)[1])          # the row through the highest MD density
    y0 = int(np.argmax(np.abs(nm - nb).mean(axis=0)))               # the row with the largest variation
    yb = 55 - 46 * r
    vmin, vmax = min(nm.min(), nf.min()) / nb, max(nm.max(), nf.max()) / nb
    ext = [0, L, 0, L]
    for j, (lab, arr) in enumerate((("MD", nm / nb), ("neural functional", nf / nb))):
        ax = ax_mm(19 + 40 * j, yb, 34, 34)
        im = ax.imshow(arr.T, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axhline(x[y0], color="w", lw=0.5, ls="--")
        ax.set_xticks([0, 10, 20]); ax.set_yticks([0, 10, 20])
        ax.set_xlabel(r"$x/\sigma$", labelpad=1); ax.set_ylabel(r"$y/\sigma$", labelpad=0)
        ax.set_title(lab, loc="left", pad=2)
        if j == 1:
            cb = fig.add_axes([(19 + 40 + 35) / W_, yb / H, 1.5 / W_, 34 / H]); fig.colorbar(im, cax=cb)
            cb.tick_params(labelsize=5, length=1.5, pad=1); cb.set_ylabel(r"$n_+/\bar n$", fontsize=5.5, labelpad=1)
        if j == 0:
            panel(ax, "ab"[r], x=-0.36, y=1.06)
            fig.text(4 / W_, (yb + 17) / H, f"$\\varepsilon_r$ = {e}\n$l_B$ = {lb}$\\sigma$", rotation=90, ha="center", va="center", fontsize=6.5)
    d = (nf - nm) / er
    ax = ax_mm(106, yb, 34, 34)
    im = ax.imshow(d.T, origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -4, 4), interpolation="nearest")
    ax.set_xticks([0, 10, 20]); ax.set_yticks([]); ax.set_xlabel(r"$x/\sigma$", labelpad=1)
    ax.set_title("difference / MD error", loc="left", pad=2)
    cb = fig.add_axes([(106 + 35) / W_, yb / H, 1.5 / W_, 34 / H]); fig.colorbar(im, cax=cb, ticks=[-4, -2, 0, 2, 4])
    cb.tick_params(labelsize=5, length=1.5, pad=1)
    ax = ax_mm(158, yb, 24, 34)
    ax.fill_between(x, (nm[:, y0] - 2 * er[:, y0]) / nb, (nm[:, y0] + 2 * er[:, y0]) / nb, color=BAND, lw=0)
    ax.plot(x, nm[:, y0] / nb, color=C_MD, lw=0.9, label="MD")
    ax.plot(x, pb[:, y0] / nb, color=C_PB, lw=0.8, ls=":", label="Poisson–Boltzmann")
    ax.plot(x, pc[:, y0] / nb, color=C_PAIR, lw=0.8, ls="--", label="pair closure")
    ax.plot(x, nf[:, y0] / nb, color=C_FUN, lw=0.9, label="neural functional")
    ax.set_xlim(0, L); ax.set_xlabel(r"$x/\sigma$", labelpad=1); ax.set_ylabel(r"$n_+/\bar n$ (cut)", labelpad=1)
    ax.set_ylim(0, 1.25 * max(nm[:, y0].max(), nf[:, y0].max()) / nb)
    if r == 0: fig.legend(*ax.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=4, handlelength=1.6, columnspacing=1.2, borderaxespad=0.2, fontsize=5.5)
    out[e] = dict(y_cut=float(x[y0]), rms_pull=float(np.sqrt(np.mean(d ** 2))), frac_within_2=float(np.mean(np.abs(d) < 2)))
save(fig, f"si/field2d_{TAG}")
json.dump(out, open(os.path.join(ROOT, f"paper/figures/si/field2d_{TAG}_metrics.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
