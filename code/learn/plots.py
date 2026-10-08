import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_profiles(rows, runs_by_key, out, ref_rows=None, title=""):

    rows = [r for r in rows if "n_pred" in r]
    if not rows:
        return
    ref = {(r["conc"], r["tag"]): r for r in (ref_rows or []) if "n_pred" in r}
    nc = 2
    nr = int(np.ceil(len(rows) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11, 2.6 * nr), squeeze=False)
    for ax, row in zip(axes.ravel(), rows):
        run = runs_by_key[f"c{row['conc']:g}/{row['tag']}"]
        z = np.arange(run.n.shape[1]) * (1.0 / run.n.shape[1])
        for a, (lab, col) in enumerate((("cation", "C0"), ("anion", "C3"))[:run.n.shape[0]]):
            ax.fill_between(z, run.n[a] - run.sig_n[a], run.n[a] + run.sig_n[a], color=col, alpha=0.25, lw=0)
            ax.plot(z, run.n[a], color=col, lw=1, label=f"MD {lab}")
            ax.plot(z, row["n_pred"][a], color=col, lw=1.4, ls="--", label=f"model {lab}")
            rr = ref.get((row["conc"], row["tag"]))
            if rr is not None:
                ax.plot(z, rr["n_pred"][a], color=col, lw=0.8, ls=":", label=f"PB {lab}")
        pm = row["profile"]
        ax.set_title(f"c={row['conc']:g} {row['tag']} {run.family}/{run.kind}  "
                     + "L2 " + "/".join(f"{pm[l]['rel_l2']:.3f}" for l in pm)
                     + f"{'' if row['el']['converged'] else '  EL NOT CONVERGED'}", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("z / L")
    axes[0, 0].legend(fontsize=7, ncol=3)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_forces(rows, runs_by_key, force_pred, out, title=""):

    rows = rows[:4]
    nsp = runs_by_key[f"c{rows[0]['conc']:g}/{rows[0]['tag']}"].n.shape[0] if rows else 2
    fig, axes = plt.subplots(len(rows), nsp, figsize=(11, 2.4 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        run = runs_by_key[f"c{row['conc']:g}/{row['tag']}"]
        fp = force_pred[f"c{row['conc']:g}/{row['tag']}"]
        z = np.arange(run.n.shape[1]) * (1.0 / run.n.shape[1])
        for a, lab in enumerate(("cation", "anion")[:nsp]):
            ax = axes[i, a]
            ax.fill_between(z, run.f[a] - run.sig_f[a], run.f[a] + run.sig_f[a], color="C0", alpha=0.3, lw=0)
            ax.plot(z, run.f[a], "C0", lw=1, label="MD f_int")
            ax.plot(z, fp[a], "k--", lw=1.2, label="model")
            ax.set_title(f"c={row['conc']:g} {row['tag']} {lab}  chi2/bin {row['chi2_' + lab]:.2f}", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_structure(struct, sps, out, refs=None, title=""):

    concs = sorted(struct)
    nsp = struct[concs[0]]["W"].shape[-1] if concs else 2
    comps = ("ZZ", "NN") if nsp == 2 else ("NN",)
    fig, axes = plt.subplots(len(comps), len(concs), figsize=(2.9 * len(concs), 2.7 * len(comps)), squeeze=False)
    for j, c in enumerate(concs):
        st, sk = struct[c], sps[round(c, 4)].sk
        n2 = nsp * float(sk["n_each"])
        for i, comp in enumerate(comps):
            ax = axes[i, j]
            ax.plot(sk["k"], sk[f"S_{comp}"] / n2, "o", ms=3, color="k", label="MD")
            kk = st["S"]["k"]
            sel = (kk > 0) & (kk <= sk["k"].max() * 1.1)
            ax.plot(kk[sel], st["S"][comp][sel] / n2, "C3-", lw=1.5, label="model")
            for name, rs in (refs or {}).items():
                if c in rs:
                    ax.plot(kk[sel], rs[c]["S"][comp][sel] / n2, lw=1, ls="--", label=name)
            if comp == "ZZ":
                kD2 = float(sk["kappaD2"])
                ax.plot(kk[sel], kk[sel] ** 2 / kD2, "0.6", lw=0.8, ls=":", label="k^2/kappa_D^2")
                ax.set_ylim(0, min(2.0, 1.3 * (sk["S_ZZ"] / n2).max()))
                ax.set_title(f"c = {c:g}   Gamma = {st['Gamma']:.2f}", fontsize=9)
            else:
                ax.set_ylim(0, 1.3 * (sk["S_NN"] / n2).max())
            ax.set_ylabel(f"S_{comp}/(2n)")
            ax.set_xlabel("k")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_Wk(struct, out, title=""):
    pairs = ((0, 0, "++"), (1, 1, "--"), (0, 1, "+-"))
    if struct and next(iter(struct.values()))["W"].shape[-1] == 1:
        pairs = ((0, 0, ""),)
    fig, axes = plt.subplots(1, len(pairs), figsize=(11, 3.2), squeeze=False); axes = axes[0]
    for c in sorted(struct):
        st = struct[c]
        k, W = st["k"], st["W"]
        for ax, (a, b, lab) in zip(axes, pairs):
            ax.plot(k, W[:, a, b], label=f"c={c:g}")
            ax.set_title(f"W_{lab}(k)")
            ax.set_xlim(0, 12)
            ax.set_xlabel("k")
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_history(hist, out, title=""):
    if hist is None or len(hist) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(hist[:, 0], hist[:, 1], label="train chi2/bin")
    ax.plot(hist[:, 0], hist[:, 2], label="held-out chi2/bin")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_scan(points, out, xlabel, title="", logx=False):

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    keys = [("heldout_chi2", "held-out chi2/bin"), ("heldout_rel_l2", "held-out rel L2 (mean)"),
            ("train_chi2", "train chi2/bin")]
    for ax, (k, lab) in zip(axes, keys):
        xs = [p[0] for p in points]
        ys = [p[2].get(k, np.nan) for p in points]
        ax.plot(xs, ys, "o-")
        for x, l, _ in points:
            ax.annotate(l, (x, ys[xs.index(x)]), fontsize=7, textcoords="offset points", xytext=(3, 3))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(lab)
        if logx:
            ax.set_xscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
