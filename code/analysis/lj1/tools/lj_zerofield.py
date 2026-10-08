#!/usr/bin/env python3
"""Zero-field analysis of one LJ state point, in the formats learn/ reads.

Streams zero_field/ions.dump (id type xu yu zu) and accumulates, per frame, the
Fourier components rho_a(k) = sum_{j in a} exp(-i k.r_j) of both labels at
k = 2 pi m / L along the three axes, m = 1..mmax.  Writes
  D.json        conc (= the density label), lz, n_pairs (particles per label), volume
  Sk.npz        S_ab(k) per volume on the axis shells (lB = 0: no charges)
  Sk_grid.npz   the same on k = 2 pi m / L with block errors (tools/sk_partial_grid.py format)
  gamma.json    Gamma(k_1) = 2 nbar / S_NN(k_1) = 1 / S(k_1), error from blocking
                and the integrated autocorrelation time of rho_N(k_1)
    lj_zerofield.py --zf <dir> --rho 0.4 [--mmax 27] [--nblock 10]
"""
import argparse
import json
import os

import numpy as np


def frames(path):
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline(); n = int(fh.readline())
            fh.readline(); box = [list(map(float, fh.readline().split()[:2])) for _ in range(3)]
            cols = fh.readline().split()[2:]
            a = np.loadtxt([fh.readline() for _ in range(n)])
            yield step, np.array(box), cols, a


def tau_int(x):
    """Integrated autocorrelation time (in frames) with the usual self-consistent window."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    t = 1.0
    for w in range(1, n):
        t += 2 * ac[w]
        if w >= 5 * t:
            break
    return max(t, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zf", required=True)
    ap.add_argument("--rho", type=float, required=True, help="density label of the state point")
    ap.add_argument("--mmax", type=int, default=27)
    ap.add_argument("--nblock", type=int, default=10)
    a = ap.parse_args()
    rk, steps = [], []
    for step, box, cols, d in frames(os.path.join(a.zf, "ions.dump")):
        L = box[0, 1] - box[0, 0]
        typ = d[:, cols.index("type")].astype(int)
        pos = d[:, [cols.index(c) for c in ("xu", "yu", "zu")]]
        k = 2 * np.pi * np.arange(1, a.mmax + 1) / L
        fr = np.zeros((3, 2, a.mmax), complex)
        for ax in range(3):
            ph = np.exp(-1j * pos[:, ax][:, None] * k[None, :])
            fr[ax, 0] = ph[typ == 1].sum(0)
            fr[ax, 1] = ph[typ == 2].sum(0)
        rk.append(fr); steps.append(step)
        N = len(typ); nh = int((typ == 1).sum())
    rk = np.array(rk)                                    # (F, 3, 2, m)
    V = L ** 3
    Spp = (np.abs(rk[:, :, 0]) ** 2).mean(1) / V         # (F, m), averaged over the axes
    Smm = (np.abs(rk[:, :, 1]) ** 2).mean(1) / V
    Spm = (rk[:, :, 0] * np.conj(rk[:, :, 1])).real.mean(1) / V
    k = 2 * np.pi * np.arange(1, a.mmax + 1) / L
    n_each = nh / V
    out = dict(m=np.arange(1, a.mmax + 1), k=k, L=L, n_frames=len(steps), n_each=n_each)
    for lab, S in (("S_pp", Spp), ("S_mm", Smm), ("S_pm", Spm)):
        bm = np.array([b.mean(0) for b in np.array_split(S, a.nblock)])
        out[lab] = S.mean(0); out[lab + "_err"] = bm.std(0, ddof=1) / np.sqrt(a.nblock)
    np.savez(os.path.join(a.zf, "Sk_grid.npz"), **out)
    SNN = out["S_pp"] + out["S_mm"] + 2 * out["S_pm"]
    SZZ = out["S_pp"] + out["S_mm"] - 2 * out["S_pm"]
    np.savez(os.path.join(a.zf, "Sk.npz"), n_each=n_each, lB=0.0, kappaD2=0.0, ratio_kmin=0.0, k=k,
             n2=np.arange(1, a.mmax + 1) ** 2, S_ZZ=SZZ, S_NN=SNN, S_pp=out["S_pp"], S_mm=out["S_mm"], S_pm=out["S_pm"])
    json.dump(dict(conc=a.rho, lz=L, n_pairs=nh, n_total=N, volume=V, rho=N / V, n_frames=len(steps),
                   steps=[int(steps[0]), int(steps[-1])]), open(os.path.join(a.zf, "D.json"), "w"), indent=1)
    # Gamma(k_1): S_NN(k_1) per frame, averaged over the axes
    x = (np.abs(rk[:, :, 0, 0] + rk[:, :, 1, 0]) ** 2).mean(1) / V / (2 * n_each)       # S_NN/(2 nbar) = S(k_1)
    ti = tau_int(x)
    err_ac = x.std(ddof=1) * np.sqrt(2 * ti / len(x))
    bm = np.array([b.mean() for b in np.array_split(x, a.nblock)])
    err_bl = bm.std(ddof=1) / np.sqrt(a.nblock)
    err = max(err_ac, err_bl)
    dt = (steps[1] - steps[0]) * 0.005
    g = dict(L=L, nbar=n_each, n_frames=len(steps), tau_total=(steps[-1] - steps[0]) * 0.005,
             shell1=dict(k=float(k[0]), S_NN_over_2n=float(x.mean()), err=float(err), err_blocks=float(err_bl),
                         err_autocorr=float(err_ac), tau_int_tau=float(ti * dt),
                         Gamma=float(1 / x.mean()), Gamma_err=float(err / x.mean() ** 2)))
    json.dump(g, open(os.path.join(a.zf, "gamma.json"), "w"), indent=1)
    print(f"{a.zf}: {len(steps)} frames, N {N}, rho {N / V:.4f}; S(k1) {x.mean():.3f} +- {err:.3f} "
          f"(tau_int {ti * dt:.0f} tau), Gamma(k1) {1 / x.mean():.3f}; S(k_max)/n {out['S_pp'][-1] / n_each:.3f}")


if __name__ == "__main__":
    main()
