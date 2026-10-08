"""2D test maps: MD n_+(x, y), the paper's model, a retrained variant (default kbK2, spectrum loss p = 2), each
difference to MD in units of the MD standard error, and a cut (MD band, paper, variant, pair closure); four potentials.
    python figs/field2d_variant_plot.py [kbK2]  -> bigbox/field2d_<variant>_eps{75,2}.png"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
B = "/mnt/research/MultiscaleML_group/Liyao"; S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
VAR = sys.argv[1] if len(sys.argv) > 1 else "kbK2"; VLAB = {"kbK2": "spectrum loss p = 2"}.get(VAR, VAR)
TAGS = ("p50", "p51", "p52", "p53")
for sysname, e in (("eps75", "7.5"), ("eps2", "2")):
    fig, axes = plt.subplots(len(TAGS), 6, figsize=(19, 3.4 * len(TAGS)))
    for i, tag in enumerate(TAGS):
        md = np.load(f"{S}/field2d/{sysname}_c0.04/field_{tag}/profiles2d.npz", allow_pickle=True)
        pr = np.load(f"{B}/salt_in_polymer_field2d/{sysname}/{tag}_pred.npz", allow_pickle=True)
        pv = np.load(f"{B}/salt_in_polymer_field2d/{sysname}/{tag}_pred_{VAR}.npz", allow_pickle=True)
        spec = json.load(open(f"{S}/field2d/{sysname}_c0.04/potentials/{tag}.json"))
        L = float(md["lz"]); x = md["x"]; nm, er = md["cation_n"], md["cation_n_err"]; nb = nm.mean()
        nf = np.array([pr[f"n_NF s{s}"][0] for s in range(3)]); pc = pr["n_pair closure"][0]
        vs = [k for k in pv.files if k.startswith(f"n_{VAR} s")]; nv = np.array([pv[k][0] for k in sorted(vs)])
        y0 = int(np.argmax(np.abs(nm - nb).mean(axis=0))); ext = [0, L, 0, L]
        vmin, vmax = min(nm.min(), nf[0].min(), nv[0].min()) / nb, max(nm.max(), nf[0].max(), nv[0].max()) / nb
        for j, (lab, arr) in enumerate((("MD", nm), ("paper loss", nf[0]), (VLAB, nv[0]))):
            ax = axes[i, j]; im = ax.imshow(arr.T / nb, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.axhline(x[y0], color="w", lw=0.6, ls="--"); ax.set_title(f"{tag} ({spec['family']}, {spec['kind']}): {lab}" if j == 0 else lab, fontsize=9)
            if j == 2: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(r"$n_+/\bar n$")
            if j == 0: ax.set_ylabel(r"$y/\sigma$")
        for j, (lab, arr) in enumerate((("paper loss", nf[0]), (VLAB, nv[0]))):
            ax = axes[i, 3 + j]; d = (arr - nm) / er
            im = ax.imshow(d.T, origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -4, 4), interpolation="nearest")
            ax.set_title(f"({lab} - MD) / MD error: rms {np.sqrt(np.mean(d ** 2)):.2f}", fontsize=9)
            if j == 1: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[-4, -2, 0, 2, 4])
        ax = axes[i, 5]
        ax.fill_between(x, (nm[:, y0] - 2 * er[:, y0]) / nb, (nm[:, y0] + 2 * er[:, y0]) / nb, color="0.8", lw=0, label="MD (2 sigma)")
        ax.plot(x, nm[:, y0] / nb, "k", lw=0.9, label="MD"); ax.plot(x, pc[:, y0] / nb, "--", color="#2ca02c", lw=0.9, label="pair closure")
        for lab, arr, col in (("paper loss", nf, "#1f77b4"), (VLAB, nv, "#d62728")):
            a = arr[:, :, y0] / nb; ax.fill_between(x, a.min(0), a.max(0), color=col, alpha=0.25, lw=0); ax.plot(x, a[0], color=col, lw=0.9, label=lab)
        ax.set_xlim(0, L); ax.set_xlabel(r"$x/\sigma$"); ax.set_ylabel(r"$n_+/\bar n$"); ax.set_title(f"cut at y = {x[y0]:.1f}$\\sigma$", fontsize=9)
        if i == 0: ax.legend(fontsize=7, loc="best")
        for ax in axes[i, :5]: ax.set_xlabel(r"$x/\sigma$")
    fig.suptitle(f"2D test, $\\varepsilon_r$ = {e}, $c$ = 0.04: cation density under $V(x, y)$ (bands: 3 seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); out = f"{S}/bigbox/field2d_{VAR}_{sysname}.png"; fig.savefig(out, dpi=110); print("wrote", out)
