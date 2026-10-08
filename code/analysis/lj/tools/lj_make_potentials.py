#!/usr/bin/env python3
"""The planar potentials of one LJ state point, with the generator of the ion
campaign (tools/make_potential.py; neutral: the same V on both labels), L = 24 sigma,
k_B T = 1.5:
  p00-p07 fourier (m 3-6, wavelengths 8-4 sigma), p08-p11 mixed (adds m 8-20, 3-1.2 sigma),
  p12-p17 gauss (Gaussian trains, period L/3 or L/4), p18-p21 long (m 1-2).
At rho >= 0.6 a draw is replaced (next seed) if a Gaussian well is deeper than
2.5 kT or a Fourier part exceeds 2 kT peak to peak, to keep the wells below the
freezing density (Cheng 2026 removed crystallised fields for the same reason).
    lj_make_potentials.py --rho 0.4 --outdir runs/lj_T1.5_rho0.40/potentials
"""
import argparse
import json
import os
import subprocess
import sys

MP = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer/tools/make_potential.py"
PLAN = [("fourier", 8), ("mixed", 4), ("gauss", 6), ("long", 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lz", type=float, default=24.0)
    ap.add_argument("--temp", type=float, default=1.5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    i = 0
    for fam, cnt in PLAN:
        for _ in range(cnt):
            tag = f"p{i:02d}"
            j = os.path.join(a.outdir, tag + ".json")
            if os.path.exists(j):
                sys.exit(f"{j} exists: refusing to overwrite")
            seed = 100000 * int(round(a.rho * 100)) + 100 * i
            while True:
                r = subprocess.run([sys.executable, MP, "--lz", str(a.lz), "--temp", str(a.temp), "--seed", str(seed),
                                    "--kind", "neutral", "--family", fam, "--tag", tag, "--out-json", j,
                                    "--out-lmp", os.path.join(a.outdir, tag + ".lmp")], capture_output=True, text=True)
                if r.returncode:
                    sys.exit(r.stderr)
                s = json.load(open(j))
                deep = (s.get("gauss", {}).get("amp_well_kT", 0.0) < -2.5) or (fam != "gauss" and s["pp_neutral_kT"] > 2.0)
                if a.rho < 0.6 or not deep:
                    break
                seed += 1
            g = s.get("gauss")
            print(f"{tag} {fam:7s} seed {seed}  pp {s['pp_neutral_kT']:.2f} kT  modes {[t['m'] for t in s['neutral']][:6]}"
                  + (f"  well {g['amp_well_kT']:.2f} kT w {g['width_well']:.2f}, barrier {g['amp_barrier_kT']:.2f} kT" if g else ""))
            i += 1


if __name__ == "__main__":
    main()
