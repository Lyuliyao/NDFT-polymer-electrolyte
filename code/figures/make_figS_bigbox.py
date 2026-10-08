"""SI figure: size test.  Driven-mode amplitude error against box length (1, 2, 4, 8 times the training box
along z, c = 0.04) for the neural functional (mean of three seeds) and the pair closure, four potentials
(p01 p09 p13 p15, the potentials of four training runs, repeated along the longer box), both eps_r.
Data: salt_in_polymer_bigbox/bigbox_summary_KBK2.json (figs/bigbox_summary.py KBK2 V1K in the eps2 root)."""
from figstyle import *
d = json.load(open("/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox/bigbox_summary_KBK2.json"))   # 2026-10-04: spectrum-loss models, refitted pair closure
BOX = {"7.5": [("x1", 1), ("x2", 2), ("x4", 4), ("x8", 8)], "2": [("x1 (training runs)", 1), ("x2", 2), ("x4", 4), ("x8", 8)]}
TAGS = ("p01", "p09", "p13", "p15")
W_, H = 89, 46
fig = plt.figure(figsize=(W_ * MM, H * MM))
out = {}
for i, e in enumerate(("7.5", "2")):
    ax = fig.add_axes([(10 + 44 * i) / W_, 9 / H, 33 / W_, 29 / H])
    for t in TAGS:
        xs = [n for b, n in BOX[e] if t in d[e][b]]
        nf = [np.mean(d[e][b][t]["V2_driven"]) for b, n in BOX[e] if t in d[e][b]]
        pc = [d[e][b][t]["V1_driven"] for b, n in BOX[e] if t in d[e][b]]
        ax.plot(xs, nf, "-o", color=C_FUN, ms=2.0, lw=0.7, alpha=0.85)
        ax.plot(xs, pc, "--s", color=C_PAIR, ms=1.9, lw=0.7, alpha=0.85)
        out.setdefault(e, {})[t] = {"box": xs, "neural": nf, "pair": pc}
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.minorticks_off()
    ax.set_yscale("log"); ax.set_ylim(0.5, 90); ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(["1", "3", "10", "30"])
    ax.set_xlabel(r"box length / $L$", labelpad=1); ax.set_ylabel("driven-mode error (%)", labelpad=1)
    ax.set_title(f"$\\varepsilon_r$ = {e}", loc="left", pad=2)
    if i == 0:
        h = [plt.Line2D([], [], color=C_FUN, marker="o", ms=2.0, lw=0.7), plt.Line2D([], [], color=C_PAIR, ls="--", marker="s", ms=1.9, lw=0.7)]
        ax.legend(h, ["neural functional", "pair closure"], loc="upper left", handlelength=1.6, borderaxespad=0.2, ncol=1)
    panel(ax, "ab"[i], x=-0.28, y=1.04)
save(fig, "si/bigbox")
json.dump(out, open(os.path.join(ROOT, "paper/figures/si/bigbox_metrics.json"), "w"), indent=1)
for e in out:
    a = np.array([v["neural"] for v in out[e].values()]); b = np.array([v["pair"] for v in out[e].values()])
    print(e, "neural %.1f-%.1f  pair %.1f-%.1f" % (a.min(), a.max(), b.min(), b.max()))
