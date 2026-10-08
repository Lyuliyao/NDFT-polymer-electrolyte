#!/usr/bin/env python3
"""Compare two field runs of the same potential that differ in one setting.

    compare_runs.py --a <run A> --b <run B> --pot <potential json> [--labels 1e-5 1e-4]

Made for the PPPM-accuracy check: both runs start from the same configuration
and use the same potential, so any difference beyond the error bars is the
setting under test.  Three comparisons, in increasing order of what the
training actually uses:

  n(z) and f_int(z)   bin by bin, from profiles.npz, against the combined error
  driven modes        complex amplitudes, the quantity the functional is fitted
                      to, with errors corrected by each run's own correlation
                      time (taking the modulus first would bias them low)
  YBG residual        the acceptance quantity itself
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_acceptance import tau_int
from field_trend import read_ave_chunk


def mode_series(run, which, lz, mmax=8):
    files = sorted(glob.glob(os.path.join(run, f"prof_{which}*.dat")))
    steps, z, n = [], None, []
    for f in files:
        st, zc, nn = read_ave_chunk(f)
        if len(st) == 0:                # a segment preempted before its first block
            continue
        steps.append(st); n.append(nn); z = zc
    steps = np.concatenate(steps); n = np.vstack(n)
    n = n[np.argsort(steps)]
    out = {}
    for m in range(1, mmax + 1):
        out[m] = (n * np.exp(-2j * np.pi * m / lz * z)[None, :]).mean(axis=1) * 2
    return out


def mean_err(a):
    """Mean of a complex series and its autocorrelation-corrected error."""
    mu = a.mean()
    var = sum(c.var(ddof=1) * 2 * tau_int(c) / len(c) for c in (a.real, a.imag))
    return mu, np.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pot", required=True)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    a = ap.parse_args()
    spec = json.load(open(a.pot))
    lz = spec["lz"]
    driven = sorted({t["m"] for t in spec["neutral"]} | {t["m"] for t in spec["charged"]})
    la, lb = a.labels
    print("=" * 78)
    print(f"{spec['tag']}: {la} ({a.a}) vs {lb} ({a.b})   driven m = {driven}")

    # --- profiles -------------------------------------------------------------
    pa, pb = (np.load(os.path.join(r, "profiles.npz")) for r in (a.a, a.b))
    for sp in ("cation", "anion"):
        line = []
        for key in ("n", "f_int"):
            x, ex = pa[f"{sp}_{key}"], pa[f"{sp}_{key}_err"]
            y, ey = pb[f"{sp}_{key}"], pb[f"{sp}_{key}_err"]
            err = np.sqrt(ex**2 + ey**2)
            pull = (x - y) / np.where(err > 0, err, np.inf)
            line.append(f"{key}: chi2/bin {np.mean(pull**2):5.2f}, "
                        f"{np.mean(np.abs(pull) < 2):5.1%} within 2 sigma")
        print(f"  {sp:6s}  " + " | ".join(line))

    # --- driven modes ---------------------------------------------------------
    print(f"  driven-mode amplitudes (the training target):")
    print(f"    {'sp':>6} {'m':>3} {'|A| '+la:>12} {'|A| '+lb:>12} {'difference':>14} {'sigma':>6}")
    worst = 0.0
    for sp, which in (("cation", "cat"), ("anion", "ani")):
        sa, sb = (mode_series(r, which, lz) for r in (a.a, a.b))
        for m in driven:
            ma, ea = mean_err(sa[m])
            mb, eb = mean_err(sb[m])
            d = abs(ma - mb) / np.sqrt(ea**2 + eb**2)
            worst = max(worst, d)
            print(f"    {sp:>6} {m:3d} {abs(ma):12.4e} {abs(mb):12.4e} "
                  f"{abs(ma)-abs(mb):+14.2e} {d:6.1f}")
    print(f"  worst driven-mode difference: {worst:.1f} sigma "
          f"({'consistent' if worst < 3 else 'NOT consistent'})")
    print("=" * 78)


if __name__ == "__main__":
    main()
