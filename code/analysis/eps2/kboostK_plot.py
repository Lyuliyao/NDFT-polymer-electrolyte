"""The paper's loss against the spectrum loss b_m = k_m^-2 (--kboost-mode k, p = 2): long-wavelength response ratio
(k_1/4, k_1/2 from the long boxes, k_1 from the training box), Gamma(k_1) against MD, held-out chi2 per concentration.
    python figs/kboostK_plot.py  -> bigbox/kboostK_p2.png"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R = "/mnt/research/MultiscaleML_group/Liyao"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
V2 = "joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3"
rel = json.load(open(f"{R}/salt_in_polymer_bigbox/bigbox_longmode_relax.json"))
ev = json.load(open(f"{R}/salt_in_polymer_eps2/runs/learn/interpretation/kboost_eval.json"))
GMD = json.load(open(f"{S}/paper/figures/fig3_metrics.json"))["Gamma"]
CONCS = ["0.01", "0.02", "0.04", "0.05", "0.06", "0.08"]; cs = [float(c) for c in CONCS]
MODELS = (("paper loss", "kb0", "NF", "#1f77b4"), ("spectrum loss p = 2", "kbK2", "kbK2", "#d62728"))
def mdir(e, key, s):
    root = f"{R}/salt_in_polymer_eps75" if e == "7.5" else f"{R}/salt_in_polymer_eps2"
    return f"{root}/runs/learn/{V2}_kn1softplus{'' if key == 'kb0' else '_' + key}{'_long' if e == '7.5' else ''}_s{s}"
fig, axes = plt.subplots(3, 2, figsize=(9.5, 10.5))
for j, (e, tag) in enumerate((("7.5", "eps75"), ("2", "eps2"))):
    # ---- long-wavelength response ratio
    ax = axes[0, j]
    for nbox, kk in ((4, 0.25), (2, 0.5)):
        r = rel[f"{tag} x{nbox} p30"]; rel_err = r["cat"]["errA"] / r["cat"]["A"]
        ax.add_patch(Rectangle((kk * 0.9, 1 - rel_err), kk * 0.2, 2 * rel_err, color="0.85", lw=0, zorder=0))
    for lab, key, relkey, col in MODELS:
        for drive, marker, dx, dl in (("p30", "o", 0.95, "neutral drive"), ("p31", "D", 1.05, "neutral + charged")):
            xs, ys, lo, hi = [], [], [], []
            for nbox, kk in ((4, 0.25), (2, 0.5)):
                r = rel[f"{tag} x{nbox} {drive}"]; a = np.array(r[f"model_{relkey}_cation"]) / r["cat"]["A"]
                xs.append(kk * dx); ys.append(a.mean()); lo.append(a.min()); hi.append(a.max())
            ys, lo, hi = map(np.array, (ys, lo, hi))
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=marker, color=col, ms=6, capsize=2, lw=1,
                        mfc=col if drive == "p30" else "white", label=f"{lab}, {dl}")
        rs = np.array([ev[f"{e}/kb{key}/s{s}"]["ratio"]["0.04"] for s in range(3)])
        ax.errorbar([1.0 * (0.97 if key == "kb0" else 1.03)], [rs.mean()], yerr=[[rs.mean() - rs.min()], [rs.max() - rs.mean()]],
                    fmt="s", color=col, ms=6, capsize=2, lw=1, label=f"{lab}, $k_1$ (training box, all long runs)")
    ax.axhline(1, ls=":", color="k", lw=0.8); ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1]); ax.set_xticklabels(["1/4", "1/2", "1"])
    ax.minorticks_off(); ax.set_xlim(0.19, 1.3); ax.set_ylim(0.76, 1.08); ax.set_xlabel("$k/k_1$"); ax.set_ylabel("amplitude, predicted / MD")
    ax.set_title(f"$\\varepsilon_r$ = {e}, $c$ = 0.04: long-wavelength response", fontsize=10)
    if j == 1: ax.legend(fontsize=6.5, loc="lower right", ncol=1)
    # ---- Gamma(k1) vs c
    ax = axes[1, j]
    md = np.array([GMD[e]["MD"][c][0] for c in CONCS]); mde = np.array([GMD[e]["MD"][c][1] for c in CONCS])
    ax.errorbar(cs, md, yerr=mde, fmt="ko", ms=5, capsize=2, lw=1, label="MD (zero field)", zorder=5)
    for lab, key, relkey, col in MODELS:
        g = np.array([[ev[f"{e}/kb{key}/s{s}"]["Gamma_k1"][c] for c in CONCS] for s in range(3)])
        ax.fill_between(cs, g.min(0), g.max(0), color=col, alpha=0.25, lw=0); ax.plot(cs, g.mean(0), "-", color=col, lw=1.3, label=lab)
    ax.set_xscale("log"); ax.set_xticks(cs); ax.set_xticklabels(CONCS); ax.minorticks_off(); ax.set_xlabel("$c$"); ax.set_ylabel("$\\Gamma(k_1)$")
    ax.set_title(f"$\\varepsilon_r$ = {e}: $\\Gamma(k_1)=2\\bar n/S_{{NN}}(k_1)$ (band: 3 seeds)", fontsize=10)
    if j == 0: ax.legend(fontsize=7, loc="upper left")
    # ---- held-out chi2 per c
    ax = axes[2, j]
    for lab, key, relkey, col in MODELS:
        per = []
        for s in range(3):
            m = json.load(open(mdir(e, key, s) + "/metrics.json")); p5 = json.load(open(mdir(e, key, s) + "/predict_c005/metrics.json"))
            rows = [r for r in m["heldout_rows"] if not (e == "2" and r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9)]
            per.append([np.mean([r["chi2"] for r in rows if abs(r["conc"] - c) < 1e-9]) if c != 0.05 else p5["transfer"]["chi2"] for c in cs])
        per = np.array(per)
        ax.fill_between(cs, per.min(0), per.max(0), color=col, alpha=0.25, lw=0); ax.plot(cs, per.mean(0), "-o", color=col, ms=4, lw=1.3, label=lab)
    ax.axhline(1, ls=":", color="k", lw=0.8); ax.set_xscale("log"); ax.set_xticks(cs); ax.set_xticklabels(CONCS); ax.minorticks_off()
    ax.set_xlabel("$c$"); ax.set_ylabel("held-out $\\chi^2$ per bin"); ax.set_ylim(0.4, 1.4)
    ax.set_title(f"$\\varepsilon_r$ = {e}: held-out force residual ($c$ = 0.05 untrained)", fontsize=10)
    if j == 0: ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
out = f"{S}/bigbox/kboostK_p2.png"; fig.savefig(out, dpi=160); print("wrote", out)
