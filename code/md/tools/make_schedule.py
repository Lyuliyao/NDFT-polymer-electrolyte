#!/usr/bin/env python3
"""Generate the external-potential schedule for one state point.

16 runs, in three families (production spec of 2026-09-23):

  p00-p07  long-wavelength Fourier, m in {3,4,5,6}; p06 is V_N only and p07 is
           psi only; peak-to-peak of each part drawn log-uniformly in
           [0.5, 3] k_B T so that weak and strong are equally represented;
  p08-p12  the same plus short modes from m in {8,12,16,20}, i.e. structure on
           the scale of the ion diameter and below;
  p13-p15  periodic Gaussian trains (period L/3 or L/4, width 0.5-2 sigma,
           well and barrier of 0.5-4 k_B T); p15 is a charged train whose
           cation and anion wells sit half a period apart.  A train is emitted
           as its exact Fourier series, so it stays box-periodic and keeps the
           mode representation the analysis uses.  Isolated Gaussians are
           avoided on purpose: they excite m = 1, whose transient is ten times
           longer.

--mode pilot returns the first two, which is what the c = 0.04 pilot ran.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def schedule(mode):
    """-> list of dicts with the make_potential.py options for each run."""
    F = lambda **kw: {"family": "fourier", "kind": "both", **kw}
    s = [F(pp_neutral=0.6, pp_charged=0.6),          # p00: the pilot pair, weak
         F(pp_neutral=2.8, pp_charged=2.8),          # p01: the pilot pair, strong
         F(), F(), F(), F(),                          # p02-p05: log-uniform draws
         F(kind="neutral"),                           # p06: V_N only
         F(kind="charged")]                           # p07: psi only
    s += [dict(family="mixed", kind="both") for _ in range(5)]        # p08-p12
    s += [dict(family="gauss", kind="neutral"),                        # p13
          dict(family="gauss", kind="neutral"),                        # p14
          dict(family="gauss", kind="charged", gauss_offset=True)]     # p15
    if mode == "long":
        # supplement of 2026-09-25: the box modes m = 1, 2 (and 3), which the
        # production families leave undriven; emitted with --tag-offset so that
        # they follow the existing p00..p15 (p00..p23 at the c = 0.04 pilot)
        return [dict(family="long", kind="neutral"),         # +0: V_N only, m in {1,2}
                dict(family="long", kind="neutral"),         # +1
                dict(family="long", kind="both"),            # +2: both parts, m in {1,2}
                dict(family="long", kind="both"),            # +3
                dict(family="long3", kind="both"),           # +4: m in {1,2,3}
                dict(family="long", kind="charged")]         # +5: psi only
    return s[:2] if mode == "pilot" else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lz", type=float, required=True)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--mode", choices=["pilot", "production", "long"], default="production")
    ap.add_argument("--seed0", type=int, default=770000)
    ap.add_argument("--subset", default=None,
                    help="slice of the schedule to emit, e.g. 8:16 for the five "
                         "mixed and three Gaussian runs")
    ap.add_argument("--tag-offset", type=int, default=0,
                    help="number the emitted runs from here, so a supplement can "
                         "be added to a state point that already has p00..p15")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    py = sys.executable
    sched = schedule(a.mode)
    lo, hi = (0, len(sched))
    if a.subset:
        lo, hi = (int(x) if x else None for x in a.subset.split(":"))
        lo, hi = lo or 0, hi if hi is not None else len(sched)
    for j, opt in enumerate(sched[lo:hi]):
        i = a.tag_offset + j if a.tag_offset else lo + j
        tag = f"p{i:02d}"
        cmd = [py, os.path.join(HERE, "make_potential.py"),
               "--lz", repr(a.lz), "--temp", repr(a.temp), "--seed", str(a.seed0 + i),
               "--kind", opt["kind"], "--family", opt["family"], "--tag", tag,
               "--out-json", os.path.join(a.outdir, tag + ".json"),
               "--out-lmp", os.path.join(a.outdir, tag + ".lmp")]
        if opt.get("gauss_offset"):
            cmd += ["--gauss-offset"]
        for key, flag in (("pp_neutral", "--pp-neutral"), ("pp_charged", "--pp-charged")):
            if opt.get(key) is not None:
                cmd += [flag, repr(opt[key])]
        subprocess.run(cmd, check=True)
    print(f"\n{hi - lo} potentials written to {a.outdir}")


if __name__ == "__main__":
    main()
