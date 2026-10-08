#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np


def read_ave_chunk(path):

    with open(path) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        i += 1
    steps, coords, vals = [], None, []
    while i < len(lines):
        head = lines[i].split()
        if len(head) < 2:
            break
        step, nch = int(head[0]), int(head[1])
        i += 1
        if i + nch > len(lines):
            break
        rows = [lines[i + j].split() for j in range(nch)]
        if any(len(r) < 4 for r in rows):
            break
        coords = np.array([float(r[1]) for r in rows])
        vals.append([float(r[3]) for r in rows])
        steps.append(step)
        i += nch
    return np.array(steps), coords, np.array(vals)


def tau_int(x):

    x = x - x.mean()
    n = len(x)
    c = np.correlate(x, x, "full")[n - 1:]
    if c[0] <= 0:
        return 0.5
    c /= c[0]
    t = 0.5
    for k in range(1, n // 4):
        if c[k] <= 0:
            break
        t += c[k]
    return max(t, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pot", required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    a = ap.parse_args()
    spec = json.load(open(a.pot))
    lz = spec["lz"]
    driven = sorted({t["m"] for t in spec["neutral"]} | {t["m"] for t in spec["charged"]})
    print("=" * 78)
    print(f"trend check  {a.run}   driven m = {driven}")

    for sp, fn in (("cation", "prof_cat.0.dat"), ("anion", "prof_ani.0.dat")):
        path = os.path.join(a.run, fn)
        if not os.path.exists(path):
            print(f"  {sp}: no {fn}")
            continue
        steps, z, n = read_ave_chunk(path)
        t = steps * a.dt
        dt_block = t[1] - t[0]
        print(f"  {sp}: {len(t)} blocks of {dt_block:.0f} tau")
        for m in driven:
            k = 2 * np.pi * m / lz
            amp = (n * np.exp(-1j * k * z)[None, :]).mean(axis=1) * 2


            ti = max(tau_int(amp.real), tau_int(amp.imag))
            neff = len(amp) / (2 * ti)
            h = len(amp) // 2
            d = abs(amp[:h].mean() - amp[h:].mean())
            derr = np.sqrt(sum(c.var(ddof=1) * 2 * tau_int(c) / len(c)
                               for part in (amp[:h], amp[h:])
                               for c in (part.real, part.imag)))
            err = derr / np.sqrt(2)
            amp = np.abs(amp)

            sl, _ = np.polyfit(t, amp, 1)
            span = sl * (t[-1] - t[0])
            print(f"    m={m}: <A> = {amp.mean():.4e}   tau_int = {ti*dt_block:6.0f} tau   "
                  f"n_indep = {neff:5.1f}   half1-half2 = {d:+.2e} +- {derr:.1e} "
                  f"({abs(d)/derr:4.1f} sigma)   trend over run = {span:+.2e} "
                  f"({abs(span)/(amp.mean()+1e-30)*100:4.1f}% of <A>)")
    print("=" * 78)


if __name__ == "__main__":
    main()
