"""Where in k the training data live, and where the models' number-channel
kernel deviates from MD.   python -m learn.diagnose_k -> runs/learn/figures/k_diagnostic.png

Top: mean power spectrum |n_+(k)|^2 of the MD profiles (signal) against the
spectrum of their error bars (noise), per concentration: the driven modes.
Bottom: the number-channel kernel n (W_++ + W_+-)(k) of V1 and V2 against
the one implied by the MD S_NN(k) (with +/- symmetry, S_NN/2n = 1/(1 + n(W_++ + W_+-))),
with the error bars from learn.gamma_md at k_1, k_2; the models' curves are the same observable from their own S_NN(k).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import data as D
from .models import W_of_k, fine_geometry
from .evaluate import S_from_W
from .protocol import load_model, OUT

C_V1, C_V2, C_MD, INK2 = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e"
CONCS = [0.01, 0.04, 0.08, 0.12]
TRAIN = [0.01, 0.02, 0.04, 0.06, 0.08]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="v1_c001_c002_c004_c006_c008")
    ap.add_argument("--v2", default="joint_v2_L1_C4_c001_c002_c004_c006_c008_val3")
    ap.add_argument("--out", default=os.path.join(OUT, "figures", "k_diagnostic.png"))
    a = ap.parse_args()
    sps, runs = D.load_all()
    models = {"V1": load_model(os.path.join(OUT, a.v1)), "V2": load_model(os.path.join(OUT, a.v2))}
    gm = json.load(open(os.path.join(OUT, "interpretation", "gamma_md.json")))
    fig, axes = plt.subplots(2, len(CONCS), figsize=(3.3 * len(CONCS), 6.4), squeeze=False)
    for j, c in enumerate(CONCS):
        sp = sps[c]; g = D.geometry_of(sp, int(round(sp.L / 0.1)))
        ax = axes[0, j]
        rs = [r for r in runs if abs(r.conc - c) < 1e-6 and r.status == "PASS"]
        if rs:
            N = rs[0].n.shape[1]; L = sp.L
            m = np.arange(N // 2 + 1); k = 2 * np.pi * m / L
            sig = np.mean([np.abs(np.fft.rfft(r.n[0]) / N) ** 2 for r in rs], axis=0)
            noise = np.mean([np.abs(np.fft.rfft(np.random.default_rng(0).normal(size=N) * r.sig_n[0]) / N) ** 2 for r in rs], axis=0)
            sel = (m >= 1) & (m <= 30)
            ax.plot(k[sel], sig[sel] / g.nbar ** 2, "o-", ms=3, color=C_MD, lw=1, label="MD profiles, |n₊(k)|²")
            ax.plot(k[sel], noise[sel] / g.nbar ** 2, "-", color="#b0afa9", lw=1, label="their error bars")
            ax.axvspan(0, 2 * np.pi * 2.5 / L, color="#f1f0ec", zorder=0)
            ax.text(2 * np.pi * 1.5 / L, 3e-1, "m = 1, 2\nundriven", ha="center", fontsize=7.5, color=INK2)
            ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
            ax.set_title(f"c = {c:g}: {len(rs)} runs" if c in TRAIN else f"c = {c:g}: no field runs", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no field runs", ha="center", va="center", transform=ax.transAxes, color=INK2)
            ax.set_title(f"c = {c:g}: no field runs", fontsize=9)
        ax.set_xlim(0, 5.5); ax.set_xlabel("k σ")
        if j == 0:
            ax.set_ylabel("profile power / n̄²")
            ax.legend(fontsize=7, frameon=False, loc="lower left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        ax = axes[1, j]
        sk = sp.sk; n2 = 2 * float(sk["n_each"])
        kernel_md = (n2 / sk["S_NN"] - 1.0)                                    # n (W_++ + W_+-) from S_NN
        ax.plot(sk["k"], kernel_md, "o", ms=3, color=C_MD, label="from MD S_NN(k)")
        for shell, mk in (("shell1", "o"), ("shell2", "^")):
            d = gm[f"{c:g}"][shell]; s, e = d["S_NN_over_2n"], d["err"]
            ax.errorbar([d["k"]], [1 / s - 1], yerr=[[1 / s - 1 - (1 / (s + e) - 1)], [1 / (s - e) - 1 - (1 / s - 1)]], fmt=mk, color=C_MD, ms=5, capsize=3)
        gf = fine_geometry(g, 8); kk = np.asarray(gf.k)
        for lab, (cfg, ps), col, ls in (("V1", models["V1"], C_V1, "--"), ("V2", models["V2"], C_V2, "-")):
            W = np.asarray(W_of_k(ps, cfg, gf))
            S = S_from_W(W, kk, g.nbar, g.lB)["NN"]          # the same observable as the MD points
            sel = kk > 0
            ax.plot(kk[sel], 2 * g.nbar / S[sel] - 1.0, ls, color=col, lw=1.6, label=lab)
        ax.axvspan(0, 2 * np.pi * 2.5 / sp.L, color="#f1f0ec", zorder=0)
        ax.axhline(0, color="#b0afa9", lw=0.6)
        ax.set_xlim(0, 3.0); ax.set_xlabel("k σ")
        ymax = max(3, 1.3 * max(kernel_md.max(), 1))
        ax.set_ylim(-1, ymax)
        if j == 0:
            ax.set_ylabel("2n / S_NN(k) − 1")
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("What the field data contain (top) and where the number-channel kernel of each model departs from MD (bottom)", fontsize=10.5)
    fig.tight_layout()
    out = a.out
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
