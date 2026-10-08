#!/usr/bin/env python3
"""Aperiodic potentials for the long boxes (size test, part C): not a repetition of a training-box potential,
and no wavelength longer than the training box.

p60 (both):    V_N = irregular Gaussian features (wells and barriers of different depths, widths and spacings)
                     + Fourier modes at wavenumbers between the training ones;  psi = Fourier modes
p61 (both):    V_N = Fourier modes;  psi = irregular charged Gaussian wells (cation wells = anion barriers)
Every component is a cosine series of the long box L = n L_small (the representation the MD, the analysis and the
functional already use).  Modes with k < k_1 = 2 pi / L_small (m < n) are removed from the Gaussian series, so that
all wavenumbers lie in the trained range [k_1, 6 k_1] (the Gaussian tails above that are the same as in the training
Gaussian trains); the removed part is reported.  Amplitudes stay inside the training range (V_N up to 4.5 kT
peak to peak, psi up to 3 kT).
    make_bigbox_aperiodic.py --lsmall L --n 2 --outdir <state>/potentials --seed 60
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_potential import draw_part, merge_modes, profile, rescale
from make_bigbox_potentials import write


def gaussians(L, feats, mmin, mmax=120, tol=1e-10):
    """Cosine series of sum_j A_j exp(-(z - z_j)^2 / 2 w_j^2) on the periodic box L; modes m < mmin dropped.
    Returns (modes, removed): the kept series and the peak-to-peak of the dropped part."""
    kept, dropped = [], []
    amax = max(abs(a) for _, a, _ in feats)
    for m in range(1, mmax + 1):
        k = 2 * np.pi * m / L
        C = sum(a * w * np.sqrt(2 * np.pi) * np.exp(-0.5 * (k * w) ** 2) * np.exp(-1j * k * z0) for z0, a, w in feats) * 2 / L
        if abs(C) < tol * amax:
            continue
        t = dict(m=m, A=float(abs(C)), k=float(k), phase=float(np.angle(C)))
        (kept if m >= mmin else dropped).append(t)
    z = np.linspace(0, L, 20001)
    return kept, float(np.ptp(profile(dropped, z))) if dropped else 0.0


def irregular(rng, L, n_feat, amps, widths, min_gap):
    """n_feat features at positions at least min_gap apart (periodic), with amplitudes and widths drawn from the ranges."""
    while True:
        z = np.sort(rng.uniform(0, L, n_feat))
        gaps = np.diff(np.concatenate([z, [z[0] + L]]))
        if gaps.min() >= min_gap:
            break
    return [(float(zj), float(rng.choice(amps) * np.exp(rng.uniform(-0.3, 0.3))), float(np.exp(rng.uniform(np.log(widths[0]), np.log(widths[1])))))
            for zj in z]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsmall", type=float, required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--outdir", required=True); ap.add_argument("--seed", type=int, default=60)
    a = ap.parse_args()
    L = a.n * a.lsmall; merge_modes.lz = L; rng = np.random.default_rng(a.seed)
    k1 = 2 * np.pi / a.lsmall
    between = [m for m in range(a.n, 6 * a.n + 1) if m % a.n]          # wavenumbers between the training ones, k_1..6 k_1
    trained = [m for m in range(a.n, 6 * a.n + 1) if m % a.n == 0]     # the training wavenumbers themselves
    z = np.linspace(0, L, 20001); pp = lambda ms: float(np.ptp(profile(ms, z)))
    os.makedirs(a.outdir, exist_ok=True)
    # p60: V_N = irregular wells and barriers + modes between the training wavenumbers; psi = modes at training wavenumbers
    feats = irregular(rng, L, 5, amps=[-2.0, -1.2, 1.5, -2.2, 1.0], widths=(0.6, 1.4), min_gap=4.0)
    g, rem = gaussians(L, feats, a.n)
    vn = merge_modes(g + draw_part(rng, L, 1.0, 4, allowed=between))
    if pp(vn) > 4.5: vn = rescale(vn, L, 4.5)
    psi = draw_part(rng, L, 1.5, 3, allowed=trained)
    spec = dict(tag="p60", seed=a.seed, kind="both", family="bigbox_aperiodic", lz=L, temp=1.0,
                pp_neutral_kT=pp(vn), pp_charged_kT=pp(psi), neutral=vn, charged=psi, size_factor=a.n,
                features=[dict(z=z0, amp_kT=am, width=w) for z0, am, w in feats], removed_pp_kT=rem,
                note=f"irregular Gaussian features (5, {rem:.2f} kT pp of modes below k_1 removed) + modes between the training wavenumbers; psi: modes at training wavenumbers")
    write(spec, a.outdir)
    # p61: V_N = modes (training and in-between wavenumbers); psi = irregular charged wells
    vn = draw_part(rng, L, 2.0, 4, allowed=between + trained)
    feats = irregular(rng, L, 4, amps=[-2.0, -1.5, -2.4, -1.2], widths=(0.7, 1.2), min_gap=5.0)
    g, rem = gaussians(L, feats, a.n)
    psi = merge_modes(g)
    if pp(psi) > 3.0: psi = rescale(psi, L, 3.0)
    spec = dict(tag="p61", seed=a.seed + 1, kind="both", family="bigbox_aperiodic", lz=L, temp=1.0,
                pp_neutral_kT=pp(vn), pp_charged_kT=pp(psi), neutral=vn, charged=psi, size_factor=a.n,
                features=[dict(z=z0, amp_kT=am, width=w) for z0, am, w in feats], removed_pp_kT=rem,
                note=f"V_N: modes at and between the training wavenumbers; psi: irregular charged Gaussian wells (4, {rem:.2f} kT pp of modes below k_1 removed)")
    write(spec, a.outdir)
    for tag in ("p60", "p61"):
        s = json.load(open(os.path.join(a.outdir, tag + ".json")))
        ms = sorted({t["m"] for t in s["neutral"] + s["charged"] if t["A"] >= 1e-3 * max(x["A"] for x in s["neutral"] + s["charged"])})
        print(f"{tag}: pp V_N {s['pp_neutral_kT']:.2f} kT, pp psi {s['pp_charged_kT']:.2f} kT; {len(s['neutral'])}+{len(s['charged'])} terms; "
              f"driven m {ms[0]}..{ms[-1]} ({len(ms)} modes, k {2*np.pi*ms[0]/L/k1:.2f}..{2*np.pi*ms[-1]/L/k1:.2f} k_1); removed below k_1: {s['removed_pp_kT']:.2f} kT pp")


if __name__ == "__main__":
    main()
