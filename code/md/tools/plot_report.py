#!/usr/bin/env python3
"""Diagnostic figures for a state point: equilibration, dynamics, structure,
and the YBG check of one field run."""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import dumpio

C = {"cation": "#1f77b4", "anion": "#d62728", "aux": "#555555", "fit": "#ff7f0e"}


def fig_npt(nptdir, ax0, ax1, discard=0.3):
    _, v = dumpio.read_ave_time(os.path.join(nptdir, "vol.dat"))
    step, L, P = v[:, 0], v[:, 1], v[:, 4]
    t = step * M.DT
    n0 = int(discard * len(v))
    ax0.plot(t, L, lw=0.7, color=C["aux"])
    ax0.axhline(L[n0:].mean(), color=C["fit"], lw=1.4, label=f"mean {L[n0:].mean():.4f}")
    ax0.axvspan(t[0], t[n0], color="k", alpha=0.07, label="discarded")
    ax0.set_xlabel(r"$t\,[\tau]$"); ax0.set_ylabel(r"$L\,[\sigma]$")
    ax0.set_title("NPT at $p=0$"); ax0.legend(fontsize=7)
    ax1.plot(t, P, lw=0.5, color=C["aux"])
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.axhline(P[n0:].mean(), color=C["fit"], lw=1.4, label=f"mean {P[n0:].mean():+.4f}")
    ax1.set_xlabel(r"$t\,[\tau]$"); ax1.set_ylabel(r"$P\,[\varepsilon/\sigma^3]$")
    ax1.set_title("pressure"); ax1.legend(fontsize=7)


def fig_msd(zfdir, ax0, ax1, res):
    t = m = None
    for lab in ("cation", "anion"):
        f = os.path.join(zfdir, f"msd_{lab}.dat")
        if not os.path.exists(f):
            continue
        d = np.loadtxt(f)
        t, m = d[1:, 0], d[1:, 1]
        ax0.loglog(t, m, color=C[lab], lw=1.2, label=lab)
        D = res.get(f"D_{lab}")
        if D:
            ax0.loglog(t, 6 * D * t, color=C[lab], lw=0.8, ls=":",
                       label=f"$6Dt$, $D={D:.2e}$")
    if t is not None:
        ax0.set_xlabel(r"$t\,[\tau]$"); ax0.set_ylabel(r"MSD $[\sigma^2]$")
        ax0.set_title("ion mean-square displacement"); ax0.legend(fontsize=7)

    f = os.path.join(zfdir, "msd_charge.dat")
    if os.path.exists(f):
        d = np.loadtxt(f)
        t, mq, ms = d[1:, 0], d[1:, 1], d[1:, 2]
        ax1.semilogx(t, mq / ms, color=C["aux"], lw=1.2)
        r = res.get("sigma_over_NE")
        if r:
            ax1.axhline(r, color=C["fit"], lw=1.4,
                        label=rf"$\sigma/\sigma_{{NE}}={r:.3f}$")
        ax1.axhline(1.0, color="k", lw=0.8, ls="--", label="fully dissociated")
        ax1.set_ylim(0, 1.6)
        ax1.set_xlabel(r"$t\,[\tau]$"); ax1.set_ylabel(r"$\sigma/\sigma_{NE}$")
        ax1.set_title("conductivity vs. Nernst-Einstein"); ax1.legend(fontsize=7)


def fig_sk(zfdir, ax, res):
    f = os.path.join(zfdir, "sk.dat")
    if not (os.path.exists(f) and res):
        return
    d = np.loadtxt(f)
    k, szz, snn = d[:, 0], d[:, 1], d[:, 2]
    n = res["n_each"]; kD2 = res["kappa_D"] ** 2
    ax.plot(k, szz / (2 * n), "o-", ms=3, color=C["cation"], lw=1.0, label=r"$S_{ZZ}/2n$")
    ax.plot(k, snn / (2 * n), "s-", ms=3, color=C["anion"], lw=1.0, label=r"$S_{NN}/2n$")
    kk = np.linspace(0, k.max(), 200)
    ax.plot(kk, kk ** 2 / kD2, "k--", lw=1.0, label=r"Debye $k^2/\kappa_D^2$")
    ax.set_xlim(0, min(k.max(), 4)); ax.set_ylim(0, 2.0)
    ax.set_xlabel(r"$k\,[\sigma^{-1}]$"); ax.set_ylabel("$S(k)/2n$")
    ax.set_title("charge structure factor"); ax.legend(fontsize=7)


def fig_field(npz, axes):
    z = npz["cation_z"]
    for lab, ax in zip(("cation", "anion"), axes[:2]):
        n, ne, V = npz[f"{lab}_n"], npz[f"{lab}_n_err"], npz[f"{lab}_V"]
        ax.errorbar(z, n, yerr=ne, color=C[lab], lw=1.0, elinewidth=0.4,
                    label=r"$n_\alpha(z)$")
        ax.axhline(n.mean(), color="k", lw=0.6, ls=":")
        ax.set_xlabel(r"$z\,[\sigma]$"); ax.set_ylabel(r"$n_\alpha\,[\sigma^{-3}]$")
        a2 = ax.twinx()
        a2.plot(z, V, color=C["aux"], lw=0.9, ls="--")
        a2.set_ylabel(r"$V_\alpha(z)\,[k_BT]$", color=C["aux"])
        ax.set_title(f"{lab}: density and applied potential")
        ax.legend(fontsize=7, loc="upper left")
    ax = axes[2]
    for lab in ("cation", "anion"):
        ax.plot(z, npz[f"{lab}_ybg_lhs"], color=C[lab], lw=1.0,
                label=rf"{lab} $k_BT\partial_z n$")
        ax.plot(z, npz[f"{lab}_ybg_rhs"], color=C[lab], lw=0.9, ls="--",
                label=rf"{lab} $f^{{tot}}$")
    ax.set_xlabel(r"$z\,[\sigma]$"); ax.set_ylabel("YBG, both sides")
    ax.set_title("Yvon-Born-Green identity"); ax.legend(fontsize=6, ncol=2)
    ax = axes[3]
    for lab in ("cation", "anion"):
        ax.plot(z, npz[f"{lab}_ybg_pull"], color=C[lab], lw=0.8, label=lab)
    for s in (-2, 2):
        ax.axhline(s, color="k", lw=0.7, ls=":")
    ax.set_ylim(-8, 8)
    ax.set_xlabel(r"$z\,[\sigma]$"); ax.set_ylabel(r"(lhs$-$rhs)/error")
    ax.set_title("YBG residual in units of its error bar"); ax.legend(fontsize=7)

    # mode-by-mode YBG: this is where the driven signal actually lives
    for lab, ax in zip(("cation", "anion"), axes[4:6]):
        m = npz[f"{lab}_m"][1:]
        lhs, rhs = npz[f"{lab}_mode_lhs"][1:], npz[f"{lab}_mode_rhs"][1:]
        le, re = npz[f"{lab}_mode_lhs_err"][1:], npz[f"{lab}_mode_rhs_err"][1:]
        dr = npz[f"{lab}_mode_driven"][1:]
        w = 0.36
        ax.bar(m - w / 2, lhs, w, yerr=le, color=C[lab], alpha=0.85,
               error_kw=dict(lw=0.7), label=r"$k_BT\,k_m|n_m|$")
        ax.bar(m + w / 2, rhs, w, yerr=re, color=C["aux"], alpha=0.85,
               error_kw=dict(lw=0.7), label=r"$|f^{tot}_m|$")
        for mm in m[dr]:
            ax.axvline(mm, color=C["fit"], lw=4, alpha=0.18, zorder=0)
        ax.set_xlabel("mode $m$"); ax.set_ylabel("amplitude")
        ax.set_title(f"{lab}: YBG mode by mode (shaded = driven)")
        ax.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--json", default=None, help="analyze_zerofield JSON")
    ap.add_argument("--field", default=None, help="field run dir with profiles.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = json.load(open(a.json)) if a.json and os.path.exists(a.json) else {}
    npt, zf = os.path.join(a.state, "equil"), os.path.join(a.state, "zerofield")

    nrow = 2 + (3 if a.field else 0)
    fig, ax = plt.subplots(nrow, 2, figsize=(10, 3.4 * nrow), squeeze=False)
    if os.path.exists(os.path.join(npt, "vol.dat")):
        fig_npt(npt, ax[0, 0], ax[0, 1])
    fig_msd(zf, ax[1, 0], ax[1, 1], res)
    if a.field:
        npz = np.load(os.path.join(a.field, "profiles.npz"))
        fig_field(npz, [ax[2, 0], ax[2, 1], ax[3, 0], ax[3, 1], ax[4, 0], ax[4, 1]])
    else:
        fig_sk(zf, ax[1, 1], res)
    fig.tight_layout()
    out = a.out or os.path.join(a.state, "report.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
