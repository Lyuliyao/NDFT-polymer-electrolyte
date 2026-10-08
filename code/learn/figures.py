"""Test-set comparison figures for V1, V2, V3 (and PB): profiles, per-run
errors, and S(k).   python -m learn.figures  ->  runs/learn/figures/"""
import argparse
import json
import os

import numpy as np
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import data as D
from .models import ModelConfig, init_params, force_density
from .evaluate import el_solve, profile_metrics, bulk_structure
from .train import residual_pulls
from .protocol import load_model, OUT

MODELS = [("V1 pair kernel", "v1_c001_c002_c004_c006", "#eb6834"),
          ("V2 pair + network", "joint_v2_L1_C4_c001_c002_c004_c006_val3", "#2a78d6"),
          ("V3 network only", "joint_v3_L1_C4_c001_c002_c004_c006_val3", "#1baf7a")]
MD_COL, PB_COL, BAND = "#0b0b0b", "#8a8984", "#d8d7d2"
CONCS = [0.01, 0.02, 0.04, 0.05, 0.06]


def load_preds(name, runs):
    """Predicted profiles for the runs from heldout.npz / predict_c005/predicted.npz."""
    d = os.path.join(OUT, name)
    out = {}
    z = np.load(os.path.join(d, "heldout.npz"))
    for k in z.files:
        out[k] = z[k]
    p = os.path.join(d, "predict_c005", "predicted.npz")
    if os.path.exists(p):
        z = np.load(p)
        for k in z.files:
            out[k.replace("transfer_", "")] = z[k]
    return {r.key: out[f"c{r.conc:g}_{r.tag}"] for r in runs if f"c{r.conc:g}_{r.tag}" in out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sps, runs = D.load_all()
    ho = D.heldout_tags(runs)
    test = [r for r in runs if r.status == "PASS" and (r.tag in ho.get(r.conc, []) and r.conc != 0.05 or r.conc == 0.05)]
    test = [r for r in test if r.conc in CONCS]
    test.sort(key=lambda r: (r.conc, r.tag))
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CONCS}

    models = {}
    for lab, name, col in MODELS:
        cfg, ps = load_model(os.path.join(OUT, name))
        models[lab] = (cfg, ps, col, load_preds(name, test))
    pb = (ModelConfig("pb"), init_params(ModelConfig("pb")))
    pb_pred = {}
    for r in test:
        n, info = el_solve(pb[1], pb[0], geoms[r.conc], r.V)
        pb_pred[r.key] = n

    # per-run metrics for every model (profile rel L2 and force chi2, mean of species)
    met = {lab: {} for lab in list(models) + ["PB"]}
    for lab, (cfg, ps, col, pred) in models.items():
        for c in CONCS:
            rs = [r for r in test if r.conc == c]
            b, _ = D.make_batches(rs, sps)[c]
            pulls = np.asarray(residual_pulls(ps, cfg, b))
            for i, r in enumerate(rs):
                pm = profile_metrics(pred[r.key], r.n)
                met[lab][r.key] = dict(chi2=float(np.mean(pulls[i] ** 2)),
                                       l2=0.5 * (pm["cation"]["rel_l2"] + pm["anion"]["rel_l2"]))
    for c in CONCS:
        rs = [r for r in test if r.conc == c]
        b, _ = D.make_batches(rs, sps)[c]
        pulls = np.asarray(residual_pulls(pb[1], pb[0], b))
        for i, r in enumerate(rs):
            pm = profile_metrics(pb_pred[r.key], r.n)
            met["PB"][r.key] = dict(chi2=float(np.mean(pulls[i] ** 2)),
                                    l2=0.5 * (pm["cation"]["rel_l2"] + pm["anion"]["rel_l2"]))
    json.dump(met, open(os.path.join(a.out, "test_metrics.json"), "w"), indent=1)

    # ---------------- figure 1: profiles, the most demanding held-out run per concentration
    fig, axes = plt.subplots(len(CONCS), 2, figsize=(12, 2.6 * len(CONCS)))
    for i, c in enumerate(CONCS):
        rs = [r for r in test if r.conc == c]
        r = max(rs, key=lambda x: x.contrast[0])
        z = (np.arange(r.n.shape[1]) + 0.5) * geoms[c].dz
        for a_, lab_sp in enumerate(("cation", "anion")):
            ax = axes[i, a_]
            ax.fill_between(z, r.n[a_] - 2 * r.sig_n[a_], r.n[a_] + 2 * r.sig_n[a_], color=BAND, lw=0, label="MD ± 2σ")
            ax.plot(z, r.n[a_], color=MD_COL, lw=1.6, label="MD")
            ax.plot(z, pb_pred[r.key][a_], color=PB_COL, lw=1.2, ls=":", label="Poisson–Boltzmann")
            for lab, (cfg, ps, col, pred) in models.items():
                ax.plot(z, pred[r.key][a_], color=col, lw=1.4, ls="--", label=lab)
            ax.set_yscale("log")
            ax.set_xlim(0, geoms[c].L)
            tag = "never trained on this concentration" if c == 0.05 else "held-out run"
            ax.set_title(f"c = {c:g}, {r.tag} ({r.family}, {r.kind}), {tag} — {lab_sp}", fontsize=9, loc="left")
            ax.grid(color="#eeeeea", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == len(CONCS) - 1:
                ax.set_xlabel("z / σ")
            if a_ == 0:
                ax.set_ylabel("n(z) / σ⁻³")
    axes[0, 0].legend(fontsize=8, ncol=3, loc="lower center", frameon=False)
    fig.suptitle("Euler–Lagrange density profiles from V(z) alone, against MD, on runs no model was trained on", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "test_profiles.png"), dpi=140)
    plt.close(fig)

    # ---------------- figure 2: every test run, profile error and force residual, all models
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True)
    x = np.arange(len(test))
    order = [("PB", PB_COL, "o"), ("V1 pair kernel", "#eb6834", "s"), ("V2 pair + network", "#2a78d6", "o"), ("V3 network only", "#1baf7a", "^")]
    for ax, key, ylab, ref in ((axes[0], "l2", "profile error, relative L2 (mean of species)", None),
                               (axes[1], "chi2", "force residual χ² per bin", 1.0)):
        for lab, col, mk in order:
            ax.plot(x, [met[lab][r.key][key] for r in test], mk, color=col, ms=6, mfc=col if lab != "PB" else "white",
                    mec=col, lw=0, label=lab, alpha=0.95)
        ax.set_yscale("log")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(axis="y", color="#eeeeea", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ref:
            ax.axhline(ref, color="#b0afa9", lw=0.8, ls="--")
            ax.text(len(test) - 0.5, ref * 1.15, "MD noise level", ha="right", fontsize=8, color="#52514e")
        # concentration groups
        prev = None
        for i, r in enumerate(test):
            if r.conc != prev:
                if i > 0:
                    ax.axvline(i - 0.5, color="#d8d7d2", lw=0.8)
                prev = r.conc
    for c in CONCS:
        idx = [i for i, r in enumerate(test) if r.conc == c]
        axes[0].text(np.mean(idx), axes[0].get_ylim()[1] * 0.75, f"c = {c:g}" + ("\n(never trained)" if c == 0.05 else "\n(held-out)"),
                     ha="center", va="top", fontsize=8.5, color="#52514e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r.tag for r in test], fontsize=7, rotation=90)
    axes[0].legend(fontsize=8.5, ncol=4, loc="lower left", frameon=False)
    fig.suptitle("Every test run: 16 held-out runs at c = 0.01–0.06 and all 16 runs at c = 0.05", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "test_errors.png"), dpi=140)
    plt.close(fig)

    # ---------------- figure 3: S(k) at every concentration, three models
    concs_all = sorted(sps)
    fig, axes = plt.subplots(2, len(concs_all), figsize=(2.6 * len(concs_all), 5.2), squeeze=False)
    for j, c in enumerate(concs_all):
        sp = sps[c]; sk = sp.sk; n2 = 2 * float(sk["n_each"]); g = D.geometry_of(sp, int(round(sp.L / 0.1)))
        for i, comp in enumerate(("ZZ", "NN")):
            ax = axes[i, j]
            ax.plot(sk["k"], sk[f"S_{comp}"] / n2, "o", ms=2.8, color=MD_COL, label="MD", zorder=5)
            for lab, (cfg, ps, col, pred) in models.items():
                st = bulk_structure(ps, cfg, g)
                kk = st["S"]["k"]; sel = (kk > 0) & (kk <= 2.2)
                ax.plot(kk[sel], st["S"][comp][sel] / n2, color=col, lw=1.4, label=lab)
            stpb = bulk_structure(pb[1], pb[0], g); kk = stpb["S"]["k"]; sel = (kk > 0) & (kk <= 2.2)
            ax.plot(kk[sel], stpb["S"][comp][sel] / n2, color=PB_COL, lw=1, ls=":", label="Poisson–Boltzmann")
            ax.set_ylim(0, 1.3 * (sk[f"S_{comp}"] / n2).max())
            ax.set_xlabel("k σ")
            ax.grid(color="#eeeeea", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"S_{comp}(k) / 2n")
            if i == 0:
                role = "trained" if c in (0.01, 0.02, 0.04, 0.06) else "not trained"
                ax.set_title(f"c = {c:g} ({role})", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Zero-field structure factors from the bulk Hessian of each functional, against MD", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "test_structure.png"), dpi=140)
    plt.close(fig)

    # a compact table
    print(f"{'model':20s} " + " ".join(f"{'c='+str(c):>14s}" for c in CONCS) + "   (profile rel L2 mean | chi2 mean)")
    for lab in ["PB"] + list(models):
        cells = []
        for c in CONCS:
            rs = [r for r in test if r.conc == c]
            cells.append(f"{np.mean([met[lab][r.key]['l2'] for r in rs]):6.3f} | {np.mean([met[lab][r.key]['chi2'] for r in rs]):5.2f}")
        print(f"{lab:20s} " + " ".join(f"{x:>14s}" for x in cells))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
