"""One figure that makes the V1 -> V2 case: the same pair kernel plus a
pointwise nonlinearity, judged where the two differ.

    python -m learn.present  ->  runs/learn/figures/v2_vs_v1.png (+ .json)

Panels: (a) force residual of the five-concentration fits on every test run,
(b, c) leave-one-concentration-out prediction, residual and profile error,
(d) Gamma(c) against MD with error bars, (e) S_NN(k) at the two ends of the
range, (f) the anion profile of the c = 0.01 held-out run V1 misses most.
"""
import argparse
import json
import os

import numpy as np
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import data as D
from .models import ModelConfig, init_params, W_of_k
from .evaluate import profile_metrics, bulk_structure
from .train import residual_pulls
from .protocol import load_model, OUT

C_V1, C_V2, C_MD, C_PB, BAND, INK2 = "#eb6834", "#2a78d6", "#0b0b0b", "#8a8984", "#d8d7d2", "#52514e"
TRAIN = [0.01, 0.02, 0.04, 0.06, 0.08]
TEST_C = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]


def ctag(cs):
    return "_".join(f"c{c:g}".replace(".", "") for c in cs)


def style(ax):
    ax.grid(color="#eeeeea", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=f"v1_{ctag(TRAIN)}")
    ap.add_argument("--v2", default=f"joint_v2_L1_C4_{ctag(TRAIN)}_val3")
    ap.add_argument("--out", default=os.path.join(OUT, "figures", "v2_vs_v1.png"))
    ap.add_argument("--loco-suffix", default="", help="name suffix of the leave-one-out transfer experiments to use")
    a = ap.parse_args()
    sps, runs = D.load_all()
    ho = D.heldout_tags(runs)
    test = [r for r in runs if r.status == "PASS" and r.conc in TEST_C
            and (r.conc == 0.05 or r.tag in ho.get(r.conc, []))]
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in sorted(sps)}
    models = {"V1": (C_V1, *load_model(os.path.join(OUT, a.v1))), "V2": (C_V2, *load_model(os.path.join(OUT, a.v2)))}
    numbers = {}

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # (a) force residual on every test run, five-concentration fits
    preds = {}
    for lab, (col, cfg, ps) in models.items():
        pr = {}
        d = os.path.join(OUT, a.v1 if lab == "V1" else a.v2)
        for f in ("heldout.npz", os.path.join("predict_c005", "predicted.npz")):
            fp = os.path.join(d, f)
            if os.path.exists(fp):
                z = np.load(fp)
                for k in z.files:
                    pr[k.replace("transfer_", "")] = z[k]
        preds[lab] = pr
        per_c = {}
        for c in TEST_C:
            rs = [r for r in test if r.conc == c]
            b, _ = D.make_batches(rs, sps)[c]
            pulls = np.asarray(residual_pulls(ps, cfg, b))
            chi = pulls.reshape(len(rs), -1) ** 2
            chi = chi.reshape(len(rs), 2, -1).mean(axis=2).mean(axis=1)
            per_c[c] = chi
            xi = TEST_C.index(c) + (-0.18 if lab == "V1" else 0.18)
            jit = np.random.default_rng(1).uniform(-0.08, 0.08, len(rs))
            ax_a.plot(xi + jit, chi, "o", ms=3.5, color=col, alpha=0.45, lw=0)
        ax_a.plot([TEST_C.index(c) + (-0.18 if lab == "V1" else 0.18) for c in TEST_C], [per_c[c].mean() for c in TEST_C],
                  "-" if lab == "V2" else "--", marker="s" if lab == "V1" else "o", ms=8, color=col, lw=1.6, label=f"{lab}, mean")
        numbers[f"test_chi2_{lab}"] = {f"{c:g}": float(per_c[c].mean()) for c in TEST_C}
    ax_a.axhline(1.0, color="#b0afa9", lw=0.9, ls="--")
    ax_a.text(-0.3, 1.08, "MD noise level", fontsize=8, color=INK2)
    ax_a.set_yscale("log")
    ax_a.set_xticks(range(len(TEST_C))); ax_a.set_xticklabels([f"{c:g}" + ("\n(never\ntrained)" if c == 0.05 else "") for c in TEST_C])
    ax_a.set_xlabel("c_LJ"); ax_a.set_ylabel("force residual χ² per bin")
    ax_a.set_title("(a) fitted on 0.01–0.08 jointly; tested on held-out runs\n      and on c = 0.05, which neither model saw", fontsize=9.5, loc="left")
    ax_a.legend(fontsize=8.5, frameon=False, loc="upper right")
    style(ax_a)

    # (b, c) leave-one-concentration-out
    loco = {"V1": {}, "V2": {}}
    for t in TRAIN:
        others = [c for c in TRAIN if c != t]
        for lab in ("V1", "V2"):
            d = os.path.join(OUT, f"transfer_{lab.lower()}_L1_C4_{ctag(others)}_to_{ctag([t])}_val3{a.loco_suffix}")
            f = os.path.join(d, "metrics.json")
            if os.path.exists(f):
                m = json.load(open(f))["transfer"]
                loco[lab][t] = (m["chi2"], 0.5 * (m["cation_rel_l2"] + m["anion_rel_l2"]))
    numbers["loco"] = {lab: {f"{t:g}": v for t, v in d.items()} for lab, d in loco.items()}
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    for lab, col, mk, ls in (("V1", C_V1, "s", "--"), ("V2", C_V2, "o", "-")):
        ts = sorted(loco[lab])
        if ts:
            xs = [TRAIN.index(t) for t in ts]
            ax_b.plot(xs, [loco[lab][t][0] for t in ts], ls, marker=mk, ms=8, color=col, lw=1.6, label=lab)
            ax_c.plot(xs, [100 * loco[lab][t][1] for t in ts], ls, marker=mk, ms=8, color=col, lw=1.6, label=lab)
    for ax, ylab, ttl in ((ax_b, "force residual χ² per bin", "(b) leave one concentration out: train on the other four,\n      predict every run at the left-out one"),
                          (ax_c, "profile error, relative L2 (%)", "(c) same, error of the Euler–Lagrange density profiles")):
        ax.set_yscale("log")
        ax.set_xticks(range(len(TRAIN))); ax.set_xticklabels([f"{c:g}" for c in TRAIN])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.set_xlabel("left-out concentration"); ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=9.5, loc="left")
        ax.legend(fontsize=8.5, frameon=False)
        style(ax)
    ax_b.axhline(1.0, color="#b0afa9", lw=0.9, ls="--")
    ax_c.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 6, 8]))
    ax_c.yaxis.set_minor_locator(FixedLocator([]))
    ax_b.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20]))
    ax_b.yaxis.set_minor_locator(FixedLocator([]))

    # (d) Gamma(c)
    gm = json.load(open(os.path.join(OUT, "interpretation", "gamma_md.json")))
    cs_md = sorted(float(c) for c in gm)
    ax_d.errorbar(cs_md, [gm[f"{c:g}"]["shell1"]["Gamma"] for c in cs_md], yerr=[gm[f"{c:g}"]["shell1"]["Gamma_err"] for c in cs_md],
                  fmt="o", color=C_MD, ms=5, capsize=3, label="MD, 2n/S_NN(k₁)", zorder=5)
    # the models are evaluated at the same wavevector as MD (k_1 = 2 pi / L), not at k = 0:
    # the field data drive k >= 0.75 only, and the box has nothing below k_1 either
    from .models import fine_geometry
    from .evaluate import S_from_W
    cs_all = sorted(sps)
    for lab, (col, cfg, ps) in models.items():
        gam_k1 = []
        for c in cs_all:
            gf = fine_geometry(geoms[c], 8); k = np.asarray(gf.k)
            S = S_from_W(np.asarray(W_of_k(ps, cfg, gf)), k, geoms[c].nbar, geoms[c].lB)["NN"] / (2 * geoms[c].nbar)
            gam_k1.append(1.0 / S[np.argmin(abs(k - 2 * np.pi / geoms[c].L))])
        ax_d.plot(cs_all, gam_k1, "-" if lab == "V2" else "--", marker="o" if lab == "V2" else "s", ms=7, color=col, lw=1.6, label=f"{lab}, 2n/S_NN(k₁)")
        numbers[f"Gamma_k1_{lab}"] = {f"{c:g}": float(v) for c, v in zip(cs_all, gam_k1)}
    ax_d.axvspan(0.01, 0.08, color="#f1f0ec", zorder=0, label="training range")
    ax_d.axhline(1, color="#b0afa9", lw=0.8, ls=":")
    ax_d.set_xscale("log"); ax_d.set_xticks([0.01, 0.02, 0.04, 0.08, 0.12]); ax_d.set_xticklabels(["0.01", "0.02", "0.04", "0.08", "0.12"])
    ax_d.set_xlabel("c_LJ"); ax_d.set_ylabel("Γ = 2n / S_NN(k₁)")
    ax_d.set_ylim(0, 16)
    ax_d.set_title("(d) long-wavelength salt-density fluctuations, model and MD at the\n      same k₁ = 2π/L" + (" (training set includes the m = 1, 2 runs)" if a.loco_suffix else "; the field data only drive k ≥ 0.75"), fontsize=9.5, loc="left")
    ax_d.legend(fontsize=8.5, frameon=False, loc="upper left")
    style(ax_d)

    # (e) S_NN(k) at the two ends
    for c, mk in ((0.01, "o"), (0.08, "^")):
        sk = sps[c].sk; n2 = 2 * float(sk["n_each"])
        ax_e.plot(sk["k"], sk["S_NN"] / n2, mk, ms=3.5, color=C_MD, mfc=C_MD if c == 0.01 else "white", label=f"MD, c = {c:g}")
        for lab, (col, cfg, ps) in models.items():
            st = bulk_structure(ps, cfg, geoms[c]); kk = st["S"]["k"]; sel = (kk > 0) & (kk <= 2.1)
            ax_e.plot(kk[sel], st["S"]["NN"][sel] / n2, "-" if lab == "V2" else "--", color=col, lw=1.6,
                      label=f"{lab}, c = {c:g}" if c == 0.01 else None)
    ax_e.set_xlabel("k σ"); ax_e.set_ylabel("S_NN(k) / 2n")
    ax_e.set_ylim(0, 1.32)
    ax_e.set_title("(e) salt-density structure factor at the two ends of the range\n      (c = 0.01 filled, c = 0.08 open; lines as in (d))", fontsize=9.5, loc="left")
    ax_e.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    style(ax_e)

    # (f) the anion profile V1 misses most at c = 0.01
    rs = [r for r in test if r.conc == 0.01]
    worst = max(rs, key=lambda r: profile_metrics(preds["V1"][f"c0.01_{r.tag}"], r.n)["anion"]["rel_l2"])
    r = worst; z = (np.arange(r.n.shape[1]) + 0.5) * geoms[0.01].dz
    ax_f.fill_between(z, r.n[1] - 2 * r.sig_n[1], r.n[1] + 2 * r.sig_n[1], color=BAND, lw=0, label="MD ± 2σ")
    ax_f.plot(z, r.n[1], color=C_MD, lw=1.6, label="MD")
    for lab, (col, cfg, ps) in models.items():
        pm = profile_metrics(preds[lab][f"c0.01_{r.tag}"], r.n)["anion"]
        ax_f.plot(z, preds[lab][f"c0.01_{r.tag}"][1], "-" if lab == "V2" else "--", color=col, lw=1.6,
                  label=f"{lab}  (rel. L2 {100*pm['rel_l2']:.1f}%)")
    ax_f.set_xlabel("z / σ"); ax_f.set_ylabel("anion density n₋(z) / σ⁻³")
    ax_f.set_title(f"(f) held-out run c = 0.01 {r.tag} ({r.family}, {r.kind}): the anion,\n      where the kernel's density dependence lives", fontsize=9.5, loc="left")
    ax_f.legend(fontsize=8, frameon=False)
    style(ax_f)

    fig.suptitle("Learned pair kernel (V1) versus pair kernel + pointwise nonlinearity (V2): identical where one concentration is fitted, different across concentrations",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    json.dump(numbers, open(a.out.replace(".png", ".json"), "w"), indent=1)
    print("wrote", a.out)
    for k, v in numbers.items():
        print(k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()})


if __name__ == "__main__":
    main()
