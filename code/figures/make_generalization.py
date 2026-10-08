"""Main-text generalization figure, using archived simulations and predictions.

Run from any directory with numpy and matplotlib installed. No model is retrained.
Only MD and the neural functional are displayed; no reference closures are loaded.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/generalization"
DATA = OUT / "data/aperiodic2d"
MM = 1 / 25.4
WIDTH, HEIGHT = 183, 106
BLUE = "#2a78d6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.labelsize": 6.5, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM))

def axes(x, y, w, h):
    return fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])

profile_file = OUT / "data/bigbox/profiles.npz"
profiles = np.load(profile_file, allow_pickle=False)
metrics = {"profiles": {}, "maps": {}, "sources": {}}
sources = [profile_file]
map_file = DATA / "maps.npz"
maps = np.load(map_file, allow_pickle=False)
sources.append(map_file)
metrics["remote_sources"] = json.loads(str(maps["sources"]))
metrics["remote_profile_sources"] = json.loads(str(profiles["sources"]))
profile_max = max(
    max((profiles[s + "_cation_n"] + 2 * profiles[s + "_cation_n_err"]).max(),
        profiles[s + "_pred"].max()) / profiles[s + "_cation_n"].mean()
    for s in ("eps75", "eps2")
)
profile_max = np.ceil(profile_max * 2) / 2

for x, letter, title in [(12, "A", "Density profile in a larger box"), (89, "B", "MD"),
                          (132, "C", "Neural functional")]:
    fig.text(x / WIDTH, 101 / HEIGHT, letter, fontsize=9, weight="bold", va="center")
    fig.text((x + 5) / WIDTH, 101 / HEIGHT, title, fontsize=7, va="center")

for row, (system, eps) in enumerate((("eps75", "7.5"), ("eps2", "2"))):
    y = 61 - 49 * row
    ax = axes(12, y, 62, 30)
    z, n, err, pred = (profiles[system + "_" + k]
                       for k in ("cation_z", "cation_n", "cation_n_err", "pred"))
    length = float(profiles[system + "_lz"])
    assert z.shape == n.shape == err.shape == pred.shape
    assert all(np.isfinite(a).all() for a in (z, n, err, pred))
    assert np.all(err >= 0) and np.all(np.diff(z) > 0)
    assert np.isclose(len(z) * np.diff(z).mean(), length)
    nb = float(n.mean())
    ax.fill_between(z, (n - 2 * err) / nb, (n + 2 * err) / nb,
                    color="0.8", lw=0)
    ax.plot(z, n / nb, color="0.15", lw=0.8, label="MD")
    ax.plot(z, pred / nb, color=BLUE, ls="--", lw=1.0, label="Neural functional")
    ax.set_xlim(0, length)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(0, profile_max)
    ax.set_xlabel(r"$z/\sigma$", labelpad=2)
    ax.set_ylabel(r"$n_+/\bar n$", labelpad=2)
    ax.set_title(rf"$\varepsilon_r = {eps}$", loc="left", pad=3)
    ax.text(0.98, 0.96, rf"$L_z=2L={length:.1f}\sigma$", transform=ax.transAxes,
            ha="right", va="top", fontsize=6)
    if row == 0:
        ax.legend(loc="lower right", bbox_to_anchor=(1.03, 1.005),
                  ncol=2, borderaxespad=0, columnspacing=0.8, handlelength=1.8)
    metrics["profiles"][eps] = {
        "potential": "p62 (mode at k_1/2 + aperiodic in-range structure)", "species": "cation", "seed": {"7.5": 0, "2": 1}[eps], "model": "KBK2",
        "box_replication": 2, "box_length": length, "bins": len(z),
        "mean_md_density": nb, "uncertainty": "plus/minus 2 archived MD standard errors",
        "processing": "raw full-box profiles; no smoothing or folding",
        "relative_l2_error": float(np.linalg.norm(pred - n) / np.linalg.norm(n)),
        "relative_mean_density_difference": float(pred.mean() / nb - 1),
        "density_axis_limits": [0, profile_max],
    }
    nm, sigma, nf = (maps[system + "_" + k] for k in ("cation_n", "cation_n_err", "pred"))
    assert nm.shape == nf.shape == sigma.shape
    assert np.isfinite(nm).all() and np.isfinite(nf).all() and np.all(sigma > 0)
    nb = float(nm.mean())
    length = float(maps[system + "_lz"])
    density_max = np.ceil(max(nm.max(), nf.max()) / nb * 2) / 2
    extent = [0, length, 0, length]
    for col, (x, data) in enumerate(((89, nm / nb), (132, nf / nb))):
        ax = axes(x, y, 30, 30)
        im = ax.imshow(data.T, origin="lower", extent=extent, interpolation="nearest",
                       cmap="viridis", vmin=0, vmax=density_max, rasterized=True)
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
        ax.set_xlabel(r"$x/\sigma$", labelpad=2)
        if col == 0:
            ax.set_ylabel(r"$y/\sigma$", labelpad=1)
        else:
            ax.set_yticklabels([])
            cb = fig.colorbar(im, cax=axes(165, y, 1.7, 30))
            cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1.5)
            cb.set_label(r"$n_+/\bar n$", fontsize=6, labelpad=2)
    difference = (nf - nm) / sigma
    metrics["maps"][eps] = {
        "potential": "p55 (aperiodic: oblique plane waves + irregular Gaussian wells and barriers)", "species": "cation", "seed": {"7.5": 0, "2": 1}[eps],
        "shape": list(nm.shape), "mean_md_density": nb, "density_color_limits": [0, density_max],
        "binwise_rms_difference_in_md_error_units": float(np.sqrt(np.mean(difference**2))),
    }

for p in sources:
    metrics["sources"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"generalization.{ext}", dpi=400)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics["maps"], indent=2))
