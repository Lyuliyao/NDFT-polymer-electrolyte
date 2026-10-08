"""SI figure: the number-density response at and below the smallest wavenumber of the training box, c = 0.04.
Ratio of the predicted to the MD amplitude of the driven m = 1 mode: at k = k_1 from the long-wavelength training
runs (sum over the runs at c = 0.04, si/long_wavelength*_metrics.json), at k_1/2 and k_1/4 from the long boxes
(bigbox_longmode_relax.json: MD amplitude = fitted equilibrium value, model = the reported model, figstyle.SEED), neutral drive (p30) and neutral + charged drive (p31).
    python make_figS_longmode.py"""
from figstyle import *
import matplotlib
B = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox"
rel = json.load(open(f"{B}/bigbox_longmode_relax.json"))
lw = {"7.5": json.load(open(os.path.join(ROOT, "paper/figures/si/long_wavelength_metrics.json"))),
      "2": json.load(open(os.path.join(ROOT, "paper/figures/si/long_wavelength_eps2_metrics.json")))}
W_, H = 89, 52
fig = plt.figure(figsize=(W_ * MM, H * MM)); out = {}
for i, (e, key) in enumerate((("7.5", "eps75"), ("2", "eps2"))):
    ax = fig.add_axes([(10 + 44 * i) / W_, 9 / H, 33 / W_, 27 / H]); o = {}
    r1 = lw[e]["m1_amplitude_by_conc"]["0.04"]
    ax.plot([1.0], [r1["fun"]], "o", color=C_FUN, ms=3.0, zorder=4); ax.plot([1.0], [r1["pair"]], "s", color=C_PAIR, ms=2.6, mfc="white", mew=0.7, zorder=4)
    o["k1"] = {"fun": r1["fun"], "pair": r1["pair"]}
    for nbox, x in ((2, 0.5), (4, 0.25)):
        # p62 / p63 (2026-10-06): the same k_1/2 mode with aperiodic in-range structure superposed (open symbols)
        for tag, dx, mk in ((("p30", -0.05, "o"), ("p31", 0.05, "D"), ("p62", -0.16, "o"), ("p63", 0.16, "D")) if nbox == 2 else (("p30", -0.03, "o"), ("p31", 0.03, "D"))):
            row = rel.get(f"{key} x{nbox} {tag}"); sup = tag in ("p62", "p63")
            if not row or "model_kbK2_cation" not in row or "model_pairKN_cation" not in row: continue      # 2026-10-04: the paper's model is the spectrum-loss one (kbK2), pair closure refitted (pairK)
            A, eA = row["cat"]["A"], row["cat"]["errA"]; nf = np.array([row["model_kbK2_cation"][SEED[e]]]) / A; pc = row["model_pairKN_cation"][0] / A   # 2026-10-06: the reported model only
            ax.plot([x * (1 + dx)], [nf.mean()], mk, color=C_FUN, ms=3.0, zorder=4, mfc="white" if sup else C_FUN, mew=0.8)
            ax.plot([x * (1 + dx)], [pc], "s", color=C_PAIR, ms=2.6, mfc="white", mew=0.7, zorder=3)
            ax.fill_between([x * (1 + dx) * 0.93, x * (1 + dx) * 1.07], 1 - eA / A, 1 + eA / A, color=BAND, lw=0, zorder=1)
            o[f"k1/{nbox} {tag}"] = {"fun": float(nf.mean()), "fun_range": [float(nf.min()), float(nf.max())], "pair": float(pc), "md_rel_err": float(eA / A)}
    ax.axhline(1, color="#b0afa9", lw=0.6, ls=":"); ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1]); ax.set_xticklabels(["1/4", "1/2", "1"]); ax.minorticks_off()
    ax.set_ylim(0.7, 1.12); ax.set_xlim(0.19, 1.3); ax.set_xlabel(r"$k/k_1$", labelpad=1); ax.set_ylabel("amplitude, predicted / MD", labelpad=1)
    ax.set_title(f"$\\varepsilon_r$ = {e}", loc="left", pad=2); panel(ax, "ab"[i], x=-0.28, y=1.04)
    if i == 0:
        h = [plt.Line2D([], [], color=C_FUN, marker="o", ms=3.0, lw=0), plt.Line2D([], [], color=C_FUN, marker="D", ms=3.0, lw=0), plt.Line2D([], [], color=C_FUN, marker="o", ms=3.0, lw=0, mfc="white", mew=0.8), plt.Line2D([], [], color=C_PAIR, marker="s", ms=2.6, mfc="white", mew=0.7, lw=0), matplotlib.patches.Patch(color=BAND, lw=0)]
        fig.legend(h, ["neural functional, neutral drive", "neural functional, neutral + charged", "open: with aperiodic structure", "pair closure", "MD error"], loc="upper left", bbox_to_anchor=(10 / W_, 0.995), ncol=2, handlelength=1.4, columnspacing=1.2, borderaxespad=0, fontsize=5)
    out[e] = o
import matplotlib
save(fig, "si/longmode"); json.dump(out, open(os.path.join(ROOT, "paper/figures/si/longmode_metrics.json"), "w"), indent=1)
for e in out: print(e, {k: (round(v["fun"], 3), round(v["pair"], 3)) for k, v in out[e].items()})
