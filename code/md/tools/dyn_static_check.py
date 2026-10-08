#!/usr/bin/env python3
"""t = 0 consistency of the dynamical data set: C(0) = kT H^-1 with the frozen functional.

    JAX_PLATFORMS=cpu <heat python> dyn_static_check.py [--model DIR] [--runs LIST] [--out FILE]

Field runs: at the measured equilibrium profile n_eq(z) (profiles.npz), the Gaussian fluctuation
covariance of the bin densities at fixed particle numbers is
    Cov = Q (Q^T J Q)^-1 Q^T / (A dz),   J_ij = d mu_i / d n_j,   mu = kT ln n + e phi[n] + mu_theta[n],
Q an orthonormal basis of profiles with zero integral for each species.  In z modes (bin centres,
divided by sinc(k dz/2) per mode so that they compare with the particle-exact modes):
    C_th^+(m, m') = V <dn_m dn_m'^*>,   C_th^-(m, m') = V <dn_m dn_m'>
against Cp, Cm at lag 0 of dyn_C.npz (block errors).  Zero-field runs: the same with n uniform,
i.e. the functional's S(k), against F_ab(k, 0) of dyn_F.npz.
kT = 1 (the half-step temperature would raise the ideal term by 0.5-0.8 %).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

ROOT = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
sys.path.insert(0, ROOT)
from learn.protocol import load_model  # noqa: E402
from learn.geometry import make_geometry  # noqa: E402
from learn.models import mu_int  # noqa: E402

MODEL = ("/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps75/runs/learn/"
         "joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3_kn1softplus_long_s0")
HELD = json.load(open(os.path.join(ROOT, "splits/heldout.json")))


def state_info(state_dir):
    if os.path.basename(state_dir.rstrip("/")) == "relax_T1.0_c0.04":      # the 64 000 tau chain: pilot state point
        state_dir = os.path.join(ROOT, "runs/pilot_T1.0_c0.04_one_both")
    zf = os.path.join(state_dir, "zero_field")
    if not os.path.isdir(zf):
        zf = os.path.join(state_dir, "zerofield")
    lB = float(np.load(os.path.join(zf, "Sk.npz"))["lB"])
    conc = float(state_dir.rstrip("/").split("_c")[-1].split("_")[0])
    return zf, lB, conc


def covariance(params, cfg, n, geom):
    N = n.shape[1]
    mu = lambda x: jnp.log(x) + mu_int(params, cfg, x, geom)                       # noqa: E731
    J = np.asarray(jax.jacfwd(mu)(jnp.asarray(n))).reshape(2 * N, 2 * N)
    J = 0.5 * (J + J.T)
    P = np.eye(2 * N)
    for s in range(2):
        P[s * N:(s + 1) * N, s * N:(s + 1) * N] -= 1.0 / N
    Q, _ = np.linalg.qr(P)
    Q = Q[:, :2 * N - 2]
    return Q @ np.linalg.inv(Q.T @ J @ Q) @ Q.T / (geom.A * geom.dz)


def modes_of(Cov, L, N, M):
    dz = L / N
    z = (np.arange(N) + 0.5) * dz
    k = 2 * np.pi * np.arange(1, M + 1) / L
    U = np.exp(-1j * np.outer(k, z)) / N / np.sinc(k * dz / 2 / np.pi)[:, None]   # [M, N]
    Ub = np.zeros((2 * M, 2 * N), complex)
    Ub[:M, :N], Ub[M:, N:] = U, U
    V = L ** 3
    Cp = V * Ub @ Cov @ Ub.conj().T
    Cm = V * Ub @ Cov @ Ub.T
    return Cp, Cm


def blocks_to_mat(C):
    """[B, a, b, m, m'] -> [B, 2M, 2M]"""
    B, _, _, M, _ = C.shape
    return C.transpose(0, 1, 3, 2, 4).reshape(B, 2 * M, 2 * M)


def compare(th, md_blocks):
    md = md_blocks.mean(0)
    se_re = md_blocks.real.std(0, ddof=1) / np.sqrt(len(md_blocks))
    se_im = md_blocks.imag.std(0, ddof=1) / np.sqrt(len(md_blocks))
    iu = np.triu_indices(md.shape[0])
    pr = ((th.real - md.real) / np.maximum(se_re, 1e-30))[iu]
    off = iu[0] != iu[1]                                  # diagonal imaginary parts are zero by construction
    pi = ((th.imag - md.imag) / np.maximum(se_im, 1e-30))[iu][off]
    p = np.concatenate([pr, pi])
    return {"rel_frob": float(np.linalg.norm(th - md) / np.linalg.norm(md)),
            "chi2_per_dof": float(np.mean(p ** 2)), "n": int(len(p)),
            "diag_ratio": (np.diag(th).real / np.diag(md).real).tolist(),
            "diag_pull": ((np.diag(th).real - np.diag(md).real) / np.diag(se_re)).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs/dyn_dataset/field_runs.txt"))
    ap.add_argument("--zf", default=os.path.join(ROOT, "runs/dyn_dataset/zf_runs.txt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/dyn_dataset/static_check.json"))
    a = ap.parse_args()
    cfg, params = load_model(a.model)
    out = {"model": a.model, "field": {}, "zero_field": {}}
    runs = [os.path.join(ROOT, l.strip()) for l in open(a.runs) if l.strip()]
    runs += [os.path.join(ROOT, "runs/prod_T1.0_c0.01/field_p00")]
    for run in runs:
        f = os.path.join(run, "dyn_C.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        state = os.path.dirname(run)
        zf, lB, conc = state_info(state)
        prof = np.load(os.path.join(run, "profiles.npz"), allow_pickle=True)
        n = np.stack([prof["cation_n"], prof["anion_n"]])
        L, M = float(d["L"]), len(d["m"])
        geom = make_geometry(L, n.shape[1], int(d["N"][0]), lB, conc)
        Cov = covariance(params, cfg, n, geom)
        thp, thm = modes_of(Cov, L, n.shape[1], M)
        tag = os.path.basename(run)
        key = f"{conc:.2f}/{tag}"
        cconc = f"{conc:g}" if conc != 0.04 else "0.04"
        r = {"heldout": tag[6:] in HELD.get(f"{conc:.2f}", []) or conc == 0.05,
             "Cp": compare(thp, blocks_to_mat(d["Cp"][..., 0])),
             "Cm": compare(thm, blocks_to_mat(d["Cm"][..., 0]))}
        out["field"][key] = r
        print(f"{key:14s} held={r['heldout']!s:5s} Cp: relF {r['Cp']['rel_frob']:.3f} chi2/dof {r['Cp']['chi2_per_dof']:6.2f}"
              f" | Cm: relF {r['Cm']['rel_frob']:.3f} chi2/dof {r['Cm']['chi2_per_dof']:6.2f}"
              f" | diag ratio m1-3 {np.round(r['Cp']['diag_ratio'][:3], 2)}", flush=True)
    for line in open(a.zf):
        if not line.strip():
            continue
        zf = os.path.join(ROOT, line.strip())
        f = os.path.join(zf, "dyn_F.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        state = os.path.dirname(zf)
        _, lB, conc = state_info(state)
        L, M = float(d["L"]), len(d["m"])
        N = int(round(L / 0.1))
        nb = np.full((2, N), float(d["n_cat"]))
        geom = make_geometry(L, N, int(round(float(d["n_cat"]) * L ** 3)), lB, conc)
        thp, _ = modes_of(covariance(params, cfg, nb, geom), L, N, M)
        th = np.stack([np.diag(thp)[:M].real, np.diag(thp)[M:].real, np.diag(thp[:M, M:]).real])
        F0 = d["F"][:, :, :, 0]
        md, se = F0.mean(1), F0.std(1, ddof=1) / np.sqrt(F0.shape[1])
        out["zero_field"][f"{conc:.2f}/{os.path.basename(zf)}"] = {
            "ratio": (th / md).tolist(), "pull": ((th - md) / se).tolist(),
            "pairs": ["++", "--", "+-"]}
        print(f"zero field {conc:.2f} {os.path.basename(zf)}: S_th/S_md m=1,2,3,6,12: "
              f"++ {np.round((th / md)[0][[0, 1, 2, 5, 11]], 2)} -- {np.round((th / md)[1][[0, 1, 2, 5, 11]], 2)}", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
