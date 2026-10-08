"""Per-mode density response in the 2x boxes for the aperiodic potentials (C: p60 p61; D: p62 p63), c = 0.04, both eps_r.
For each run and species the complex amplitudes a_m = 2 <n exp(-i k_m z)> of MD (error from the 8 blocks) and of the
models (KBK2 three seeds, pair closure V1K, old loss NF), then per wavenumber band the amplitude ratio
sum |a_pred| / sum |a_MD|, the complex error sum |a_pred - a_MD| / sum |a_MD| and the share (%) of the band in the total MD\namplitude, over the driven modes of the band whose MD amplitude exceeds three block errors.
Bands in units of the training-box k_1 (= 2 k_box): [0.5,1) (the box mode, D only), [1,1.5), [1.5,2), [2,3), [3,6], (6,40].
    python figs/bigbox_modes_aperiodic.py C|D   -> prints a table, writes bigbox/x2_modes_<fam>_eps{75,2}.png"""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
FAM = sys.argv[1] if len(sys.argv) > 1 else "C"
TAGS = {"C": ("p60", "p61"), "D": ("p62", "p63")}[FAM]
BANDS = ((0.5, 0.999), (1.0, 1.499), (1.5, 1.999), (2.0, 2.999), (3.0, 6.001), (6.002, 40.0))
MODELS = (("KBK2", (0, 1, 2)), ("NF", (0, 1, 2)), ("V1K", (None,)))


def amp(n, z, L, ms):
    return np.array([2 * (n * np.exp(-2j * np.pi * m * z / L)).mean() for m in ms])


for e, sysname in (("7.5", "eps75"), ("2", "eps2")):
    root = f"{B}/{sysname}_c0.04_x2"; fig, axes = plt.subplots(len(TAGS), 2, figsize=(12, 3.2 * len(TAGS)), sharex=True)
    print(f"== eps_r = {e}: per band  ratio sum|a_pred|/sum|a_MD|  (complex error %)   bands in k/k_1(training box)")
    for i, tag in enumerate(TAGS):
        md = np.load(f"{S}/bigbox/{sysname}_c0.04_x2/field_{tag}/profiles.npz", allow_pickle=True)
        spec = json.loads(str(md["pot_json"])); L = float(md["lz"]); k1 = 2 * np.pi / (L / 2)
        terms = spec.get("neutral", []) + spec.get("charged", []); amax = max(t["A"] for t in terms)
        ms = np.array(sorted({t["m"] for t in terms if t["A"] >= 1e-3 * amax})); kk = 2 * np.pi * ms / L / k1
        preds = {}
        for key, seeds in MODELS:
            arr = []
            for s in seeds:
                f = f"{root}/runs/learn/{key}{'' if s is None else f'_s{s}'}/predict_c004/predicted.npz"
                if os.path.exists(f) and f"transfer_c0.04_{tag}" in np.load(f).files: arr.append(np.load(f)[f"transfer_c0.04_{tag}"])
            if arr: preds[key] = np.array(arr)
        for a, sp in enumerate(("cation", "anion")):
            z, n, blocks = md[f"{sp}_z"], md[f"{sp}_n"], md[f"{sp}_n_blocks"]
            am = amp(n, z, L, ms); ab = np.array([amp(b, z, L, ms) for b in blocks]); aerr = np.sqrt(((np.abs(ab) - np.abs(ab).mean(0)) ** 2).mean(0) / (len(ab) - 1))
            ax = axes[i, a]; ax.errorbar(kk, np.abs(am) / n.mean(), yerr=aerr / n.mean(), fmt="ko", ms=3, lw=0.8, label="MD")
            line = f"  {tag} {sp:6s}"; sig = np.abs(am) > 3 * aerr
            for (key, col, lab) in (("KBK2", "#d62728", "spectrum loss p = 2"), ("NF", "#1f77b4", "old loss"), ("V1K", "#2ca02c", "pair closure")):
                if key not in preds: continue
                ap = np.array([amp(p[a], z, L, ms) for p in preds[key]])
                ax.plot(kk, np.abs(ap[0]) / n.mean(), "-", color=col, lw=0.9, label=lab)
                if len(ap) > 1: ax.fill_between(kk, np.abs(ap).min(0) / n.mean(), np.abs(ap).max(0) / n.mean(), color=col, alpha=0.2, lw=0)
                cells = []
                sig = np.abs(am) > 3 * aerr           # significant MD modes only (the tails of the Gaussian series are driven at 1e-3 kT)
                for lo, hi in BANDS:
                    sel = (kk >= lo) & (kk <= hi) & sig
                    if not sel.any(): cells.append("     --     "); continue
                    r = [np.abs(x[sel]).sum() / np.abs(am[sel]).sum() for x in ap]; ce = [100 * np.abs(x[sel] - am[sel]).sum() / np.abs(am[sel]).sum() for x in ap]
                    cells.append(f"{np.mean(r):4.2f}({np.mean(ce):3.0f}%,{100 * np.abs(am[sel]).sum() / np.abs(am[sig]).sum():2.0f})")
                line += f" | {key:4s} " + " ".join(cells)
            print(line + f"   [{sig.sum()} of {len(ms)} driven modes significant]")
            ax.set_yscale("log"); ax.set_ylim(1e-3, 1); ax.set_ylabel(f"$|a_m|/\\bar n$ ({sp})"); ax.text(0.02, 0.05, tag, transform=ax.transAxes)
            for lo, _ in BANDS[1:]: ax.axvline(lo, color="0.85", lw=0.6)
            if i == len(TAGS) - 1: ax.set_xlabel("$k / k_1$ (training box)")
            if i == 0 and a == 0: ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"driven-mode amplitudes, 2x box, $\\varepsilon_r$ = {e}, c = 0.04"); fig.tight_layout()
    out = f"{S}/bigbox/x2_modes_{FAM}_{sysname}.png"; fig.savefig(out, dpi=130); print("wrote", out)
