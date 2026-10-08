#!/usr/bin/env python3

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

    from field_trend import read_ave_chunk
    import glob
    files = sorted(glob.glob(os.path.join(run, f"prof_{which}*.dat")))
    if not files:
        raise SystemExit(f"no prof_{which}.*.dat in {run}")
    steps, z, n = [], None, []
    for f in files:
        st, zc, nn = read_ave_chunk(f)
        if len(st) == 0:
            continue
        steps.append(st); n.append(nn); z = zc
    steps = np.concatenate(steps); n = np.vstack(n)


    last = {s: i for i, s in enumerate(steps)}
    keep = np.array(sorted(last.values()))
    steps, n = steps[keep], n[keep]
    order = np.argsort(steps)
    steps, n = steps[order], n[order]
    h = len(steps) // 2
    out = []
    for part in (n[:h], n[h:]):
        mean = part.mean(axis=0)

        err = np.array([p.std(ddof=1) * np.sqrt(2 * tau_int(p) / len(p)) for p in part.T])
        out.append((mean, err))
    return out[0], out[1], z, len(steps)


def mode_halves(run, which, lz, modes):

    from field_trend import read_ave_chunk
    files = sorted(glob.glob(os.path.join(run, f"prof_{which}*.dat")))
    steps, z, n = [], None, []
    for f in files:
        st, zc, nn = read_ave_chunk(f)
        if len(st) == 0:
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
    files = sorted(glob.glob(os.path.join(run, "current*.dat")))
    if not files:
        return None
    d = np.vstack([np.loadtxt(f, comments="#") for f in files])
    d = d[np.argsort(d[:, 0])]
    out = {}
    for i, lab in ((1, "cation"), (2, "anion")):
        v = d[:, i]


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
