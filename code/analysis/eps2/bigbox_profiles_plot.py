"""Density profiles in the 2x box (c = 0.04): MD (band 2 sigma) against the paper's model, the spectrum-loss p = 2 model
(3 seeds each: band + seed 0) and the pair closure, for the long modes p30 (neutral) / p31 (neutral + charged) and the
repeated training potentials p01 p09 p13 p15.    python figs/bigbox_profiles_plot.py  -> bigbox/x2_profiles_eps{75,2}.png
    python figs/bigbox_profiles_plot.py C   -> the aperiodic in-range potentials p60 p61 (pair closure V1K)  -> bigbox/x2_profiles_C_eps{75,2}.png
    python figs/bigbox_profiles_plot.py D   -> the superposed potentials p62 p63 (k1/2 mode + in-range structure)   -> bigbox/x2_profiles_D_eps{75,2}.png"""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
FAM = sys.argv[1] if len(sys.argv) > 1 else ""
TAGS = {"": (("p30", "mode k1/2, neutral"), ("p31", "mode k1/2, neutral + charged"), ("p01", "training potential p01, repeated"),
             ("p09", "p09, repeated"), ("p13", "p13, repeated"), ("p15", "p15, repeated")),
        "C": (("p60", "aperiodic: Gaussian features + modes between the training k; psi modes"), ("p61", "aperiodic: V_N modes; psi charged Gaussian wells")),
        "D": (("p62", "k1/2 (1 kT, neutral) + aperiodic structure"), ("p63", "k1/2 (1 kT, neutral + charged) + aperiodic structure"))}[FAM]
PAIR = "V1" if FAM == "" else "V1K"
MODELS = (("paper loss", "NF", "#1f77b4"), ("spectrum loss p = 2", "KBK2", "#d62728"))
for e, sysname in (("7.5", "eps75"), ("2", "eps2")):
    root = f"{B}/{sysname}_c0.04_x2"; fig, axes = plt.subplots(len(TAGS), 2, figsize=(13, 2.3 * len(TAGS)), sharex=True)
    for i, (tag, title) in enumerate(TAGS):
        md = np.load(f"{S}/bigbox/{sysname}_c0.04_x2/field_{tag}/profiles.npz", allow_pickle=True)
        z = md["cation_z"]; L = len(z) * (z[1] - z[0])
        for a, sp in enumerate(("cation", "anion")):
            ax = axes[i, a]; n, er = md[f"{sp}_n"], md[f"{sp}_n_err"]; nb = n.mean()
            ax.fill_between(z, (n - 2 * er) / nb, (n + 2 * er) / nb, color="0.8", lw=0, label="MD (2 sigma)")
            ax.plot(z, n / nb, color="k", lw=0.8, label="MD")
            v1 = np.load(f"{root}/runs/learn/{PAIR}/predict_c004/predicted.npz")[f"transfer_c0.04_{tag}"][a]
            ax.plot(z, v1 / nb, "--", color="#2ca02c", lw=0.9, label="pair closure")
            for lab, key, col in MODELS:
                ps = np.array([np.load(f"{root}/runs/learn/{key}_s{s}/predict_c004/predicted.npz")[f"transfer_c0.04_{tag}"][a] for s in range(3)]) / nb
                ax.fill_between(z, ps.min(0), ps.max(0), color=col, alpha=0.25, lw=0); ax.plot(z, ps[0], color=col, lw=0.9, label=lab)
            ax.set_xlim(0, L); ax.set_ylabel(f"$n_{'+' if a == 0 else '-'}/\\bar n$")
            if i == 0: ax.set_title(("cation" if a == 0 else "anion") + f"   ($\\varepsilon_r$ = {e}, $c$ = 0.04, box 2L = {L:.0f}$\\sigma$)", fontsize=10)
            ax.text(0.01, 0.95, f"{tag}: {title}", transform=ax.transAxes, fontsize=8, va="top")
            if i == len(TAGS) - 1: ax.set_xlabel("$z/\\sigma$")
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97)); out = f"{S}/bigbox/x2_profiles_{FAM + '_' if FAM else ''}{sysname}.png"; fig.savefig(out, dpi=140); print("wrote", out)
