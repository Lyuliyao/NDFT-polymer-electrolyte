#!/usr/bin/env python3
"""The two Section 7 checks that analyze_profiles.py does not cover.

    field_acceptance.py --run <field dir> [--binw 0.1] [--out acceptance.json]

  1. The profiles from the first and the second half of the production agree.
     Both halves come from the 500-tau time series that `fix ave/chunk` writes,
     and the error of each bin is corrected by its own integrated
     autocorrelation time.  That correction is not optional here: a bin's
     density carries the undriven long-wavelength modes, whose relaxation
     (m = 2 measures tau_int ~ 1600 tau, m = 1 longer still) is slower than any
     practical block, so a plain block error is too small by 1.5-2x and flags
     healthy runs.  A drifting run fails this test even when its YBG residual
     looks fine, because YBG holds instantaneously.

  2. The mean ion current vanishes.  The per-species centre-of-mass velocity is
     read from the current.<seg>.dat files (all segments of a resumed run) and
     compared with its own standard error.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio
import model as M


def tau_int(x):
    """Integrated autocorrelation time, in units of the sampling interval."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    c = np.correlate(x, x, "full")[n - 1:]
    if c[0] <= 0:
        return 0.5
    c = c / c[0]
    t = 0.5
    for k in range(1, n // 4):
        if c[k] <= 0:
            break
        t += c[k]
    return max(t, 0.5)


def halves(run, which):
    """Per-bin mean and autocorrelation-corrected error for each half of the
    production, from the fix ave/chunk series (500 tau per sample)."""
    from field_trend import read_ave_chunk
    import glob
    files = sorted(glob.glob(os.path.join(run, f"prof_{which}*.dat")))
    if not files:
        raise SystemExit(f"no prof_{which}.*.dat in {run}")
    steps, z, n = [], None, []
    for f in files:                      # a resumed run has one file per segment
        st, zc, nn = read_ave_chunk(f)
        if len(st) == 0:                 # preempted before its first full block
            continue
        steps.append(st); n.append(nn); z = zc
    steps = np.concatenate(steps); n = np.vstack(n)
    # a resumed segment repeats the blocks written after its checkpoint by the
    # interrupted one: keep one block per step, the later segment's
    last = {s: i for i, s in enumerate(steps)}
    keep = np.array(sorted(last.values()))
    steps, n = steps[keep], n[keep]
    order = np.argsort(steps)
    steps, n = steps[order], n[order]
    h = len(steps) // 2
    out = []
    for part in (n[:h], n[h:]):
        mean = part.mean(axis=0)
        # error per bin, inflated by that bin's own correlation time
        err = np.array([p.std(ddof=1) * np.sqrt(2 * tau_int(p) / len(p)) for p in part.T])
        out.append((mean, err))
    return out[0], out[1], z, len(steps)


def mode_halves(run, which, lz, modes):
    """Half-to-half agreement mode by mode: the complex amplitude of each mode
    is averaged over each half and the difference is compared with an error
    that corrects the real and the imaginary part by their own correlation
    time.  This is the sharp form of the test.  The bin-by-bin version above
    cannot be made sharp at this run length, because a single bin carries the
    slow undriven long-wavelength modes."""
    from field_trend import read_ave_chunk
    files = sorted(glob.glob(os.path.join(run, f"prof_{which}*.dat")))
    steps, z, n = [], None, []
    for f in files:
        st, zc, nn = read_ave_chunk(f)
        if len(st) == 0:                # a segment preempted before its first block
            continue
        steps.append(st); n.append(nn); z = zc
    steps = np.concatenate(steps); n = np.vstack(n)
    o = np.argsort(steps); n = n[o]
    h = len(n) // 2
    out = {}
    for m in modes:
        a = (n * np.exp(-2j * np.pi * m / lz * z)[None, :]).mean(axis=1) * 2
        num = a[:h].mean() - a[h:].mean()
        var = sum(c.var(ddof=1) * 2 * tau_int(c) / len(c)
                  for part in (a[:h], a[h:]) for c in (part.real, part.imag))
        out[m] = float(abs(num) / np.sqrt(var)) if var > 0 else 0.0
    return out


def current(run):
    files = sorted(glob.glob(os.path.join(run, "current*.dat")))   # current.dat or current.<seg>.dat
    if not files:
        return None
    d = np.vstack([np.loadtxt(f, comments="#") for f in files])
    d = d[np.argsort(d[:, 0])]
    out = {}
    for i, lab in ((1, "cation"), (2, "anion")):
        v = d[:, i]
        # a block error over ten blocks ignores the correlation time and comes
        # out about twice too small, which flags healthy runs
        err = v.std(ddof=1) * np.sqrt(2 * tau_int(v) / len(v))
        out[lab] = (float(v.mean()), float(err))
    return out, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--binw", type=float, default=0.1)
    ap.add_argument("--nblock", type=int, default=8)
    ap.add_argument("--pot", default=None,
                    help="potential json: gives the box edge and which modes are "
                         "driven.  Without it the test would decompose at the "
                         "wrong k and would miss the driven modes of a Gaussian "
                         "train, which reach m = 32.")
    ap.add_argument("--lz", type=float, default=None, help="overrides the box edge")
    ap.add_argument("--mmax", type=int, default=7,
                    help="highest mode in the half-to-half test; must cover the\n"
                         "driven modes of the mixed and gauss families")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"run": a.run}
    driven, lz = [], a.lz
    if a.pot:
        spec = json.load(open(a.pot))
        lz = a.lz or spec["lz"]
        driven = sorted({t["m"] for t in spec["neutral"]} | {t["m"] for t in spec["charged"]})
    if lz is None:
        raise SystemExit("need --pot or --lz: the box differs between state points")
    modes = sorted(set(driven) | set(range(1, a.mmax + 1)))
    res["driven"] = driven
    print("=" * 74)
    print(f"field acceptance  {a.run}")
    print(f"  L = {lz:.4f}, driven modes {driven}")

    ok = True
    for sp, which in (("cation", "cat"), ("anion", "ani")):
        (n1, e1), (n2, e2), zc, nframe = halves(a.run, which)
        err = np.sqrt(e1**2 + e2**2)
        pull = (n1 - n2) / np.where(err > 0, err, np.inf)
        within = float((np.abs(pull) < 2).mean())
        chi2 = float((pull**2).mean())
        res[sp] = dict(samples=nframe, half_within_2sigma=within,
                       half_chi2_per_bin=chi2, half_max_pull=float(np.abs(pull).max()))
        print(f"  {sp:6s}: halves agree on {within:5.1%} of {len(zc)} bins "
              f"(chi^2/bin {chi2:.2f}, max pull {np.abs(pull).max():.1f})   [diagnostic]")

        mp = mode_halves(a.run, which, lz, modes)
        res[sp]["mode_half_pull"] = mp
        # only the driven modes gate: they carry the signal the functional is
        # fitted to.  The undriven ones are thermal fluctuations whose slowest
        # members (m = 1, 2) hold only ~10 independent samples in a run of this
        # length, so they scatter widely and would fail healthy data.
        dv = {m: mp[m] for m in driven} or mp
        wd = max(dv, key=dv.get)
        good = dv[wd] < 3.0
        ok &= good
        line = (f"          driven modes: median {np.median(list(dv.values())):.1f} sigma, "
                f"worst m={wd} at {dv[wd]:.1f} sigma  {'OK' if good else 'REVIEW'}")
        und = {m: mp[m] for m in mp if m not in driven}
        if und:
            wu = max(und, key=und.get)
            line += f"   [undriven worst m={wu} at {und[wu]:.1f} sigma, not a gate]"
        print(line)

    c = current(a.run)
    if c:
        cur, nsamp = c
        print(f"  mean ion current over {nsamp} samples:")
        for lab, (mu, se) in cur.items():
            z = abs(mu) / se if se > 0 else 0.0
            res[f"vcm_{lab}"], res[f"vcm_{lab}_err"] = mu, se
            good = z < 3
            ok &= good
            print(f"    v_cm,z({lab:6s}) = {mu:+.3e} +- {se:.1e}  ({z:.1f} sigma from zero)  "
                  f"{'OK' if good else 'REVIEW'}")
    else:
        print("  no current.*.dat found")

    res["pass"] = bool(ok)
    print(f"  -> {'PASS' if ok else 'REVIEW'}")
    print("=" * 74)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
