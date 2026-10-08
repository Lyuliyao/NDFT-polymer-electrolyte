"""SI figure: aperiodic potentials in the twofold box (size test part C), c = 0.04, both eps_r.
Rows: density profiles under p60 (cation, anion) and p61 (cation, anion): MD with a 2-sigma band, the neural functional
(seed 0, band over three seeds) and the pair closure; bottom row: amplitude of every significant driven mode, predicted / MD,
against k / k_1 of the training box (p60 filled, p61 open; bars: the MD relative error).  Metrics (profile errors, driven-mode
errors with their MD noise level, band ratios) -> si/aperiodic_metrics.json.
    python make_figS_aperiodic.py"""
from figstyle import *
B = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox"; S = ROOT
TAGS = ("p60", "p61"); SEEDS = (0, 1, 2)


def amp(n, z, L, ms):
    return np.array([2 * (n * np.exp(-2j * np.pi * m * z / L)).mean() for m in ms])


W_, H = 183, 118
fig = plt.figure(figsize=(W_ * MM, H * MM)); out = {}
for i, (e, sysname) in enumerate((("7.5", "eps75"), ("2", "eps2"))):
    root = f"{B}/{sysname}_c0.04_x2"; x0 = (12 + 92 * i) / W_; o = {}
    pred = {m: np.load(f"{root}/runs/learn/{m}/predict_c004/predicted.npz") for m in [f"KBK2_s{s}" for s in SEEDS] + ["V1KN"]}
    for j, tag in enumerate(TAGS):
        md = np.load(f"{S}/bigbox/{sysname}_c0.04_x2/field_{tag}/profiles.npz", allow_pickle=True)
        spec = json.loads(str(md["pot_json"])); L = float(md["lz"]); k1 = 2 * np.pi / (L / 2)
        terms = spec["neutral"] + spec["charged"]; amax = max(t["A"] for t in terms)
        ms = np.array(sorted({t["m"] for t in terms if t["A"] >= 1e-3 * amax})); kk = 2 * np.pi * ms / L / k1
        ot = {"driven_modes": int(len(ms))}
        for a, sp in enumerate(("cation", "anion")):
            z, n, er, blocks = md[f"{sp}_z"], md[f"{sp}_n"], md[f"{sp}_n_err"], md[f"{sp}_n_blocks"]; nb = n.mean()
            ax = fig.add_axes([x0, (H - 12 - 17 * (2 * j + a + 1) + 3) / H, 80 / W_, 14 / H])
            ax.fill_between(z, (n - 2 * er) / nb, (n + 2 * er) / nb, color=BAND, lw=0, zorder=1)
            ax.plot(z, n / nb, color=C_MD, lw=0.7, zorder=3)
            ps = np.array([pred[f"KBK2_s{s}"][f"transfer_c0.04_{tag}"][a] for s in SEEDS]) / nb
            ax.fill_between(z, ps.min(0), ps.max(0), color=C_FUN, alpha=0.3, lw=0, zorder=2); ax.plot(z, ps[0], color=C_FUN, lw=0.7, zorder=4)
            ax.plot(z, pred["V1KN"][f"transfer_c0.04_{tag}"][a] / nb, color=C_PAIR, lw=0.7, ls="--", zorder=3)
            ax.set_xlim(0, L); ax.set_ylabel(("$n_+$" if a == 0 else "$n_-$") + r"$/\bar n$", labelpad=1)
            if a == 0: ax.set_title(f"$\\varepsilon_r$ = {e}, {tag}", loc="left", pad=2); panel(ax, "abcd"[2 * i + j], x=-0.13, y=1.02)
            if j == 1 and a == 1: ax.set_xlabel(r"$z/\sigma$", labelpad=1)
            else: ax.set_xticklabels([])
            # amplitudes
            am = amp(n, z, L, ms); ab = np.array([amp(b, z, L, ms) for b in blocks])
            aerr = np.sqrt(((np.abs(ab) - np.abs(ab).mean(0)) ** 2).mean(0) / (len(ab) - 1)); sig = np.abs(am) > 3 * aerr
            ap = np.array([amp(pred[f"KBK2_s{s}"][f"transfer_c0.04_{tag}"][a], z, L, ms) for s in SEEDS]); av = amp(pred["V1KN"][f"transfer_c0.04_{tag}"][a], z, L, ms)
            ot[sp] = dict(n_sig=int(sig.sum()), md_noise_pct=float(100 * np.sqrt(np.pi / 2) * aerr[sig].sum() / np.abs(am[sig]).sum()),
                          den=float(np.abs(am[sig]).sum()), noise_num=float(np.sqrt(np.pi / 2) * aerr[sig].sum()),
                          fun_num=[float(np.abs(x[sig] - am[sig]).sum()) for x in ap], pair_num=float(np.abs(av[sig] - am[sig]).sum()),
                          fun_err_pct=[float(100 * np.abs(x[sig] - am[sig]).sum() / np.abs(am[sig]).sum()) for x in ap],
                          pair_err_pct=float(100 * np.abs(av[sig] - am[sig]).sum() / np.abs(am[sig]).sum()),
                          ratio_k1=float(np.mean([np.abs(x[0]) for x in ap]) / np.abs(am[0])) if (sig[0] and ms[0] == 2) else None,
                          ratio_k1_md_err=float(aerr[0] / np.abs(am[0])) if (sig[0] and ms[0] == 2) else None,
                          ratio_above=float(np.mean([np.abs(x[sig & (kk > 1.2)]).sum() for x in ap]) / np.abs(am[sig & (kk > 1.2)]).sum()),
                          share_k1=float(np.abs(am[0]) / np.abs(am[sig]).sum()) if ms[0] == 2 else 0.0)
            ot[sp + "_modes"] = dict(k=kk[sig].tolist(), ratio=(np.abs(ap).mean(0)[sig] / np.abs(am[sig])).tolist(), md_rel_err=(aerr[sig] / np.abs(am[sig])).tolist(),
                                     pair=(np.abs(av[sig]) / np.abs(am[sig])).tolist())
        den = ot["cation"]["den"] + ot["anion"]["den"]
        ot["both"] = dict(fun_err_pct=[100 * (ot["cation"]["fun_num"][s] + ot["anion"]["fun_num"][s]) / den for s in range(len(SEEDS))],
                          pair_err_pct=100 * (ot["cation"]["pair_num"] + ot["anion"]["pair_num"]) / den,
                          md_noise_pct=100 * (ot["cation"]["noise_num"] + ot["anion"]["noise_num"]) / den)
        o[tag] = ot
    # bottom: ratio per mode
    ax = fig.add_axes([x0, 8 / H, 80 / W_, 24 / H])
    for j, tag in enumerate(TAGS):
        for a, sp in enumerate(("cation", "anion")):
            mm = o[tag][sp + "_modes"]; k, r, er, pc = map(np.array, (mm["k"], mm["ratio"], mm["md_rel_err"], mm["pair"]))
            col = C_CAT if a == 0 else C_ANI
            ax.errorbar(k, r, yerr=er, fmt="o" if j == 0 else "o", mfc=col if j == 0 else "white", mec=col, ecolor=col, ms=2.2, mew=0.6, lw=0.5, capsize=0, zorder=4 - j)
            ax.plot(k, pc, "s", mfc="none", mec=C_PAIR, ms=1.8, mew=0.5, zorder=2)
    ax.axhline(1, color="#b0afa9", lw=0.6, ls=":"); ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"]); ax.minorticks_off()
    ax.set_ylim(0.6, 1.4); ax.set_xlim(0.85, 24); ax.set_xlabel(r"$k/k_1$", labelpad=1); ax.set_ylabel("amplitude, predicted / MD", labelpad=1); panel(ax, "ef"[i], x=-0.13, y=1.0)
    if i == 0:
        h = [plt.Line2D([], [], color=C_CAT, marker="o", ms=2.2, lw=0), plt.Line2D([], [], color=C_ANI, marker="o", ms=2.2, lw=0),
             plt.Line2D([], [], color=C_CAT, marker="o", mfc="white", ms=2.2, lw=0), plt.Line2D([], [], color=C_PAIR, marker="s", mfc="none", ms=1.8, lw=0)]
        ax.legend(h, ["cation, p60", "anion, p60", "p61 (open)", "pair closure"], loc="upper left", ncol=4, handlelength=1.0, columnspacing=0.9, bbox_to_anchor=(0, 1.02))
    out[e] = o
hh = [plt.Line2D([], [], color=C_MD, lw=0.8), matplotlib.patches.Patch(color=BAND), plt.Line2D([], [], color=C_FUN, lw=0.8), plt.Line2D([], [], color=C_PAIR, lw=0.8, ls="--")]
fig.legend(hh, ["MD", r"MD $\pm2$ s.e.", "neural functional", "pair closure"], loc="upper left", bbox_to_anchor=(12 / W_, 0.998), ncol=4, handlelength=1.6, columnspacing=1.2)
import matplotlib
save(fig, "si/aperiodic"); json.dump(out, open(os.path.join(ROOT, "paper/figures/si/aperiodic_metrics.json"), "w"), indent=1)
for e in out:
    for tag in TAGS:
        r = out[e][tag]["both"]; print(e, tag, "both species:", f"fun {np.mean(r['fun_err_pct']):.1f} [{min(r['fun_err_pct']):.1f}-{max(r['fun_err_pct']):.1f}]  pair {r['pair_err_pct']:.1f}  noise {r['md_noise_pct']:.1f}")
        for sp in ("cation", "anion"):
            r = out[e][tag][sp]; print(e, tag, sp, f"sig {r['n_sig']}/{out[e][tag]['driven_modes']}  fun {np.mean(r['fun_err_pct']):.1f} [{min(r['fun_err_pct']):.1f}-{max(r['fun_err_pct']):.1f}]  pair {r['pair_err_pct']:.1f}  noise {r['md_noise_pct']:.1f}  "
                                    f"ratio k1 {r['ratio_k1']} +- {r['ratio_k1_md_err']}  share {100 * r['share_k1']:.1f}%  ratio >1.2k1 {r['ratio_above']:.3f}")
