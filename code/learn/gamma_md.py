"""Gamma from the zero-field trajectories with error bars.

Gamma(k) = 2 nbar / S_NN(k) at the sampled finite wavevectors.
S_NN(k) = <|rho_N(k)|^2>/V is computed frame
by frame for the k vectors of the first shells, so that its statistical error
can be taken from the time series: blocks, inflated by the integrated
autocorrelation time of |rho_N(k,t)|^2 (the k_min number mode relaxes on
1/(D k^2) ~ 1e4 tau, comparable to the run length).  Reports the first and
second half separately, and the value from the first 20 000 tau alone.

    python -m learn.gamma_md [--concs 0.04 0.06 ...]  -> runs/learn/interpretation/gamma_md.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def tau_int(x, cmax=None):
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("autocorrelation requires a finite series with at least two samples")
    x = x - x.mean()
    n = len(x)
    cmax = max(1, min(cmax or n // 4, n))
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[:cmax] / np.arange(n, n - cmax, -1)
    if c[0] <= 0:
        return 0.5
    c /= c[0]
    t = 0.5
    for k in range(1, cmax):
        if c[k] <= 0:
            break
        t += c[k]
    return max(t, 0.5)


def shell_series(pos, L, n2max=2):
    """|rho_N(k,t)|^2 / V for every k vector with 0 < |n|^2 <= n2max; dict n2 -> (nk, nt)."""
    ks = []
    r = int(np.ceil(np.sqrt(n2max)))
    for nx in range(0, r + 1):
        for ny in range(-r, r + 1):
            for nz in range(-r, r + 1):
                if nx == 0 and (ny < 0 or (ny == 0 and nz <= 0)):
                    continue
                n2 = nx * nx + ny * ny + nz * nz
                if 0 < n2 <= n2max:
                    ks.append((n2, nx, ny, nz))
    ks.sort()
    kvec = 2 * np.pi * np.array([k[1:] for k in ks], float) / L
    V = L ** 3
    nt = pos.shape[0]
    out = np.zeros((len(ks), nt))
    chunk = 2000
    for i0 in range(0, nt, chunk):
        p = pos[i0:i0 + chunk]                                  # (nc, N, 3)
        ph = np.einsum("tnd,kd->tnk", p, kvec)
        rho = np.exp(-1j * ph).sum(axis=1)                      # (nc, nk)
        out[:, i0:i0 + chunk] = (np.abs(rho) ** 2).T / V
    groups = {}
    for i, k in enumerate(ks):
        groups.setdefault(k[0], []).append(i)
    return {n2: out[idx] for n2, idx in groups.items()}, {n2: 2 * np.pi * np.sqrt(n2) / L for n2 in groups}


def shell_stat(series, nblock=10):
    """Average directions at each frame, retaining their covariance.

    Estimate block and autocorrelation errors of that same shell-mean time
    series.  The larger error is reported; it is not a convergence certificate.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 2 or not series.shape[0] or series.shape[1] < 2:
        raise ValueError("shell series must have shape (nk >= 1, nt >= 2)")
    if not np.isfinite(series).all() or nblock < 2:
        raise ValueError("finite samples and at least two blocks are required")
    s = series.mean(axis=0)
    nt = s.size
    nblock = min(nblock, nt)
    ti = tau_int(s)
    bl = np.array([b.mean() for b in np.array_split(s, nblock)])
    se_block = bl.std(ddof=1) / np.sqrt(nblock)
    se_tau = s.std(ddof=1) * np.sqrt(2 * ti / nt)
    return float(s.mean()), float(max(se_block, se_tau)), float(ti)


def main():
    from . import data as D
    from .protocol import OUT

    ap = argparse.ArgumentParser()
    ap.add_argument("--concs", type=float, nargs="+", default=None)
    ap.add_argument("--frames-first", type=int, default=20001, help="frames in the first 20000 tau")
    a = ap.parse_args()
    sps, _ = D.load_all()
    out = {}
    for c, sp in sorted(sps.items()):
        if a.concs and not any(abs(c - x) < 1e-6 for x in a.concs):
            continue
        t0 = time.time()
        z = np.load(os.path.join(sp.zero_field, "ions.dump.npz"))
        cols = list(z["cols"]); data = z["data"]; steps = z["steps"]
        ix = [cols.index(k) for k in ("x", "y", "z")] if "x" in cols else [cols.index(k) for k in ("xu", "yu", "zu")]
        pos = data[:, :, ix]
        L = float(z["boxes"][0][0, 1] - z["boxes"][0][0, 0])
        nbar = sp.n_pairs / L ** 3
        dt_frame = (steps[1] - steps[0]) * 0.005
        ser, kval = shell_series(pos, L)
        res = dict(L=L, nbar=nbar, n_frames=int(len(steps)), tau_total=float(len(steps) * dt_frame))
        for n2 in sorted(ser):
            s = ser[n2]
            m, e, ti = shell_stat(s)
            h = s.shape[1] // 2
            m1, e1, _ = shell_stat(s[:, :h]); m2, e2, _ = shell_stat(s[:, h:])
            mf, ef, _ = shell_stat(s[:, :a.frames_first])
            g = lambda mm, ee: (2 * nbar / mm, 2 * nbar / mm ** 2 * ee)
            res[f"shell{n2}"] = dict(k=kval[n2], S_NN_over_2n=m / (2 * nbar), err=e / (2 * nbar),
                                    tau_int_tau=ti * dt_frame, Gamma=g(m, e)[0], Gamma_err=g(m, e)[1],
                                    Gamma_half1=g(m1, e1)[0], Gamma_half2=g(m2, e2)[0],
                                    Gamma_first20k=g(mf, ef)[0], Gamma_first20k_err=g(mf, ef)[1])
        out[f"{c:g}"] = res
        r1, r2 = res["shell1"], res["shell2"]
        print(f"c={c:<5g} L={L:.2f} frames={len(steps)} ({res['tau_total']:.0f} tau)  "
              f"k1={r1['k']:.3f}: S_NN/2n={r1['S_NN_over_2n']:.3f}+-{r1['err']:.3f} tau_int={r1['tau_int_tau']:.0f}tau  "
              f"Gamma={r1['Gamma']:.2f}+-{r1['Gamma_err']:.2f} halves {r1['Gamma_half1']:.2f}/{r1['Gamma_half2']:.2f} "
              f"first20k {r1['Gamma_first20k']:.2f}+-{r1['Gamma_first20k_err']:.2f} | k2: Gamma={r2['Gamma']:.2f}+-{r2['Gamma_err']:.2f}  [{time.time()-t0:.0f}s]",
              flush=True)
    os.makedirs(os.path.join(OUT, "interpretation"), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "interpretation", "gamma_md.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
