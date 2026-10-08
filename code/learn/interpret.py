"""What the learned functional says in real space.

    python -m learn.interpret [--models joint_v2_L1_C4_c001_c002_c004 ...]

Writes runs/learn/interpretation/:
  pair_kernels.png   u_ab(r) of the learned pair parts, with and without the
                     mean-field Coulomb l_B e_a e_b / r
  W_r.png            the full bulk kernel W_ab(r) of the model at several
                     concentrations (3D inverse transform of W(k)), against
                     the density-independent V1 kernel
  gamma_c.png        the thermodynamic factor Gamma(c) as a continuous curve,
                     with the MD estimate 2n / S_NN(k_min) as points
  bulk.json          numbers: u_ab(0), integrals, W_ab(0), Gamma(c)
"""
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import data as D
from .geometry import make_geometry
from .models import W_of_k, fine_geometry, ModelConfig, init_params
from .protocol import load_model, OUT

PAIRS = ((0, 0, "++"), (1, 1, "--"), (0, 1, "+-"))
E = np.array([1.0, -1.0])


def gauss3d(r, s):
    return (2 * np.pi * s * s) ** -1.5 * np.exp(-0.5 * (r / s) ** 2)


def pair_kernel_r(cfg, params, r):
    """u_ab(r) = sum_m c_abm B_m(r) from the pair part; (3, nr)."""
    c = cfg.pair_scale * np.asarray(params["pair"])
    B = np.stack([gauss3d(r, s) for s in cfg.widths()])
    return c @ B


def W_r_from_k(k, Wk, r):
    """3D radial inverse transform W(r) = (2 pi^2)^-1 int k^2 W(k) j0(kr) dk."""
    kr = np.outer(r, k)
    j0 = np.where(kr > 1e-12, np.sin(kr) / np.where(kr > 1e-12, kr, 1.0), 1.0)
    return np.trapezoid(k ** 2 * Wk[None, :] * j0, k, axis=1) / (2 * np.pi ** 2)


def bulk_W(cfg, params, sp, factor=8):
    g = fine_geometry(D.geometry_of(sp, int(round(sp.L / 0.1))), factor)
    return np.asarray(g.k), np.asarray(W_of_k(params, cfg, g))


def gamma_curve(cfg, params, sp_ref, concs):
    """Gamma(c) = 1 + (n/2) sum_ab W_ab(0) with W at the uniform density of c
    (box of the reference state point; W(0) does not depend on it)."""
    out = []
    N = int(round(sp_ref.L / 0.1))
    for c in concs:
        n_pairs = c * 12000
        g = make_geometry(sp_ref.L, N, n_pairs, sp_ref.lB, c)
        W0 = np.asarray(W_of_k(params, cfg, g, ks=jnp.zeros(1)))[0]
        out.append(1.0 + 0.5 * g.nbar * W0.sum())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "v1_c004", "v1_c001_c002_c004", "joint_v2_L1_C4_c001_c002_c004",
        "joint_v3_L1_C4_c001_c002_c004", "joint_v2_L2_C4_c001_c002_c004"])
    ap.add_argument("--concs", type=float, nargs="+", default=[0.01, 0.04, 0.12])
    ap.add_argument("--train-range", type=float, nargs=2, default=[0.01, 0.06])
    a = ap.parse_args()
    out = os.path.join(OUT, "interpretation")
    os.makedirs(out, exist_ok=True)
    sps, _ = D.load_all()
    models = {n: load_model(os.path.join(OUT, n)) for n in a.models}
    lB = sps[0.04].lB
    r = np.linspace(0.0, 6.0, 601)
    numbers = {}

    # ---- learned pair kernels in real space --------------------------------
    pair_models = [n for n, (cfg, _) in models.items() if cfg.has_pair]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for n in pair_models:
        cfg, ps = models[n]
        u = pair_kernel_r(cfg, ps, r)
        c = cfg.pair_scale * np.asarray(ps["pair"])
        numbers[n] = {"u_at_0": dict(zip(["++", "--", "+-"], u[:, 0].tolist())),
                      "integral_u": dict(zip(["++", "--", "+-"], c.sum(1).tolist()))}
        for j, (ia, ib, lab) in enumerate(PAIRS):
            axes[0, j].plot(r, u[j], label=n)
            v = u[j] + lB * E[ia] * E[ib] / np.maximum(r, 0.05)
            axes[1, j].plot(r, v, label=n)
    for j, (ia, ib, lab) in enumerate(PAIRS):
        axes[0, j].set_title(f"learned u_{lab}(r)  [kT]")
        axes[0, j].axhline(0, color="0.7", lw=0.6)
        axes[1, j].plot(r, lB * E[ia] * E[ib] / np.maximum(r, 0.05), "k:", lw=0.8, label="Coulomb only")
        axes[1, j].set_title(f"u_{lab} + l_B e e / r")
        axes[1, j].set_ylim(-40, 40)
        axes[1, j].axhline(0, color="0.7", lw=0.6)
        axes[1, j].set_xlabel("r / sigma")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pair_kernels.png"), dpi=130)
    plt.close(fig)

    # ---- the full bulk kernel W(r) at several concentrations --------------
    fig, axes = plt.subplots(len(a.concs), 3, figsize=(12, 3.0 * len(a.concs)), squeeze=False)
    ref = "v1_c001_c002_c004" if "v1_c001_c002_c004" in models else pair_models[0]
    for i, c in enumerate(a.concs):
        sp = sps[round(c, 4)]
        for n, (cfg, ps) in models.items():
            if cfg.variant == "v1" and n != ref:
                continue
            k, Wk = bulk_W(cfg, ps, sp)
            numbers.setdefault(n, {}).setdefault("W0", {})[f"{c:g}"] = dict(
                zip(["++", "--", "+-"], [float(Wk[0, ia, ib]) for ia, ib, _ in PAIRS]))
            for j, (ia, ib, lab) in enumerate(PAIRS):
                Wr = W_r_from_k(k, Wk[:, ia, ib], r)
                axes[i, j].plot(r, Wr, label=n, ls="--" if cfg.variant == "v1" else "-")
        for j, (ia, ib, lab) in enumerate(PAIRS):
            axes[i, j].set_title(f"c = {c:g}   W_{lab}(r)  [kT sigma^3]", fontsize=9)
            axes[i, j].axhline(0, color="0.7", lw=0.6)
            axes[i, j].set_xlabel("r / sigma")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "W_r.png"), dpi=130)
    plt.close(fig)

    # ---- Gamma(c) ------------------------------------------------------------
    cs = np.geomspace(0.005, 0.15, 40)
    fig, ax = plt.subplots(figsize=(6, 4))
    gam_md = {}
    gmd_file = os.path.join(out, "gamma_md.json")
    if os.path.exists(gmd_file):
        # full-trajectory values with autocorrelation-corrected errors (learn.gamma_md)
        gm = json.load(open(gmd_file))
        cs_md = sorted(float(c) for c in gm)
        g1 = [gm[f"{c:g}"]["shell1"]["Gamma"] for c in cs_md]
        e1 = [gm[f"{c:g}"]["shell1"]["Gamma_err"] for c in cs_md]
        g2 = [gm[f"{c:g}"]["shell2"]["Gamma"] for c in cs_md]
        ax.errorbar(cs_md, g1, yerr=e1, fmt="ko", capsize=3, label="MD: 2n / S_NN(k_1), full run")
        ax.plot(cs_md, g2, "k^", ms=4, mfc="none", label="MD: k_2 shell")
        gam_md = dict(zip(cs_md, g1))
    else:
        for c, sp in sorted(sps.items()):
            sk = sp.sk
            gam_md[c] = float(2 * sk["n_each"] / sk["S_NN"][:2].mean())
        ax.plot(list(gam_md), list(gam_md.values()), "ko", label="MD: 2n / S_NN(k_min)")
    for n, (cfg, ps) in models.items():
        gam = gamma_curve(cfg, ps, sps[0.04], cs)
        numbers.setdefault(n, {})["Gamma_curve"] = dict(c=cs.tolist(), Gamma=gam.tolist())
        ax.plot(cs, gam, label=n)
    ax.axhline(1.0, color="0.6", ls=":", label="ideal / PB")
    ax.axvspan(a.train_range[0], a.train_range[1], color="0.9", label="training range")
    ax.set_xscale("log")
    ax.set_xlabel("c_LJ")
    ax.set_ylabel("Gamma")
    ax.set_ylim(-1, 16)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gamma_c.png"), dpi=130)
    plt.close(fig)
    numbers["Gamma_MD_estimate"] = gam_md
    json.dump(numbers, open(os.path.join(out, "bulk.json"), "w"), indent=1)

    # ---- a few numbers to the terminal -------------------------------------
    print("learned pair kernels: u(0) and integral (kT sigma^3), pairs ++ / -- / +-")
    for n in pair_models:
        u0, iu = numbers[n]["u_at_0"], numbers[n]["integral_u"]
        print(f"  {n:36s} u(0) {u0['++']:8.2f} {u0['--']:8.2f} {u0['+-']:8.2f}   "
              f"integral {iu['++']:8.1f} {iu['--']:8.1f} {iu['+-']:8.1f}")
    print("W(0) per concentration (++ / -- / +-):")
    for n, (cfg, ps) in models.items():
        if "W0" in numbers.get(n, {}):
            print(f"  {n}")
            for c, w in numbers[n]["W0"].items():
                print(f"     c={c:5s} {w['++']:8.1f} {w['--']:8.1f} {w['+-']:8.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
