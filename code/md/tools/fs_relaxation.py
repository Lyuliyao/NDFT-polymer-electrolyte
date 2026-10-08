#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio
import model as M


def fs_acf(z, k, chunk=128):

    nt, nion = z.shape
    n2 = 1 << (2 * nt - 1).bit_length()
    acc = np.zeros(nt)
    for lo in range(0, nion, chunk):
        e = np.exp(1j * k * z[:, lo:lo + chunk])
        F = np.fft.fft(e, n=n2, axis=0)
        acf = np.fft.ifft(F * np.conjugate(F), axis=0)[:nt]
        acc += acf.real.sum(axis=1)
    norm = (nt - np.arange(nt)) * nion
    return acc / norm


def tau_from(fs, dt):

    ok = fs > np.exp(-3.0)
    n = int(ok.sum())
    if n < 5:
        return float("nan"), float("nan")
    t = np.arange(n) * dt

    m = (fs[:n] < 0.8) & (fs[:n] > np.exp(-3.0))
    tau_fit = float("nan")
    if m.sum() >= 5:
        sl = np.polyfit(t[m], np.log(fs[:n][m]), 1)[0]
        tau_fit = -1.0 / sl if sl < 0 else float("nan")
    integral = np.trapz(fs[:n], t)
    if np.isfinite(tau_fit):
        integral += tau_fit * fs[n - 1]
    return float(integral), tau_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--modes", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    steps, boxes, cols, data = dumpio.read_dump(os.path.join(a.run, "ions.dump"),
                                                stride=a.stride)
    typ = data[0, :, cols.index("type")].astype(int)
    zu = data[:, :, cols.index("zu")]
    del data
    dt = (steps[1] - steps[0]) * M.DT
    lz = float(boxes[0][2, 1] - boxes[0][2, 0])
    zu = zu - zu.mean(axis=1, keepdims=True)
    D = {}
    dj = os.path.join(a.run, "D.json")
    if os.path.exists(dj):
        d = json.load(open(dj))
        D = {"cation": d["D_cation"], "anion": d["D_anion"]}

    print("=" * 78)
    print(f"F_s relaxation  {a.run}   {len(steps)} frames, {steps[-1]*M.DT:.0f} tau, L={lz:.3f}")
    print(f"  {'sp':>6} {'m':>3} {'lambda':>7} {'tau_Fs':>9} {'tau_fit':>9} "
          f"{'1/(k^2 D)':>10} {'ratio':>6}")
    out = {}
    for lab, tt in (("cation", M.T_CAT), ("anion", M.T_ANI)):
        sel = typ == tt
        for m in a.modes:
            k = 2.0 * np.pi * m / lz
            fs = fs_acf(zu[:, sel], k)
            tau, tfit = tau_from(fs, dt)
            ref = 1.0 / (k * k * D[lab]) if lab in D else float("nan")
            out[f"{lab}_m{m}"] = dict(tau_Fs=tau, tau_fit=tfit, tracer=ref)
            print(f"  {lab:>6} {m:3d} {lz/m:7.2f} {tau:9.0f} {tfit:9.0f} "
                  f"{ref:10.0f} {tau/ref if ref == ref else float('nan'):6.2f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
