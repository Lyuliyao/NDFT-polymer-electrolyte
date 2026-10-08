#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def schedule(mode):

    F = lambda **kw: {"family": "fourier", "kind": "both", **kw}
    s = [F(pp_neutral=0.6, pp_charged=0.6),
         F(pp_neutral=2.8, pp_charged=2.8),
         F(), F(), F(), F(),
         F(kind="neutral"),
         F(kind="charged")]
    s += [dict(family="mixed", kind="both") for _ in range(5)]
    s += [dict(family="gauss", kind="neutral"),
          dict(family="gauss", kind="neutral"),
          dict(family="gauss", kind="charged", gauss_offset=True)]
    if mode == "long":


        return [dict(family="long", kind="neutral"),
                dict(family="long", kind="neutral"),
                dict(family="long", kind="both"),
                dict(family="long", kind="both"),
                dict(family="long3", kind="both"),
                dict(family="long", kind="charged")]
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
