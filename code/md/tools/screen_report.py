#!/usr/bin/env python3
"""Tabulate the solvation-variant screen against the paper (Figs. S1a, S2)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_reference as PR

root = sys.argv[1] if len(sys.argv) > 1 else "runs/screen_solv"
jname = sys.argv[2] if len(sys.argv) > 2 else "screen.json"
ref = PR.expected(0.04)
rows = [("ij_both", "sigma_ij, both ions (eq. 3; current)"),
        ("one_both", "sigma=1, both ions"),
        ("ij_cation", "sigma_ij, cation only"),
        ("one_cation", "sigma=1, cation only")]

print(f"paper (c_s = 0.08, T* = 1.0):  V = {ref['V']:.0f}   D+ = {ref['D_cation']:.2e}   "
      f"D- = {ref['D_anion']:.2e}   D-/D+ = {ref['D_anion']/ref['D_cation']:.2f}\n")
print(f"{'variant':38} {'V':>7} {'dV':>7} {'D+':>9} {'D-':>9} {'D-/D+':>6} "
      f"{'sig/NE':>6}  score")
for key, lab in rows:
    f = os.path.join(root, key, jname)
    if not os.path.exists(f):
        print(f"{lab:38} (not run)")
        continue
    r = json.load(open(f))
    V = r["L_mean"] ** 3
    dV = V / ref["V"] - 1
    dp = r["D_cation"] / ref["D_cation"] - 1
    dm = r["D_anion"] / ref["D_anion"] - 1
    # rms relative deviation over the three independent observables
    score = (dV ** 2 + dp ** 2 + dm ** 2) ** 0.5 / 3 ** 0.5
    print(f"{lab:38} {V:7.0f} {100*dV:+6.1f}% {r['D_cation']:9.2e} {r['D_anion']:9.2e} "
          f"{r['D_anion']/r['D_cation']:6.2f} {r['sigma_over_NE']:6.2f}  {100*score:5.1f}%")
    sl = (r.get("msd_slope_cation", 1.0), r.get("msd_slope_anion", 1.0))
    if min(sl) < 0.9:
        print(f"{'':38}   ^ MSD log-log slopes {sl[0]:.2f}/{sl[1]:.2f}: not yet diffusive, "
              f"D is a lower bound")
print("\nscore = rms relative deviation over V, D+, D-.  The SI values are "
      "digitised by eye (+-5%), and D from 2000 tau carries roughly +-10%.")
