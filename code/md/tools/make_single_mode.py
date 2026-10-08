#!/usr/bin/env python3
"""Single-mode, single-channel external potential for the drive pilots (03h, 2026-10-04).

    make_single_mode.py --m 6 --channel N|psi --A 1.3 [--phase 0] [--period 100] --lz L --out DIR/TAG

    V_N : V_+ = V_- = A cos(k z + phase)
    psi : V_+ = -V_- = A cos(k z + phase)            k = 2 pi m / L_z, peak to peak 2A (k_B T units)
With --period T the potential is multiplied by sin(2 pi t / T), t = step * dt (timestep reset to 0
at the switch-on), so it starts from zero.  Writes TAG.json and TAG.lmp; the .lmp defines the variables
in.06_relax reads for DIR = z: v_fz_cat, v_ez_cat, v_fz_ani, v_ez_ani (force = -dV/dz).
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--channel", choices=["N", "psi"], required=True)
    ap.add_argument("--A", type=float, required=True)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--period", type=float, default=0.0)
    ap.add_argument("--lz", type=float, required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k = 2 * np.pi * a.m / a.lz
    sign = {"cat": 1.0, "ani": 1.0 if a.channel == "N" else -1.0}
    spec = {"m": a.m, "k": k, "channel": a.channel, "A": a.A, "pp_kT": 2 * a.A, "phase": a.phase,
            "period_tau": a.period, "lz": a.lz, "force_amplitude": a.A * k}
    lines = [f"# single mode m={a.m} channel={a.channel} A={a.A} pp={2 * a.A:g} kT phase={a.phase:.6f}"
             + (f" period={a.period:g} tau" if a.period else " step")]
    tf = ""
    if a.period:
        lines.append(f"variable tdrive equal sin({2 * np.pi / a.period:.12g}*step*{a.dt})")
        tf = "*v_tdrive"
    for sp, s in sign.items():
        lines.append(f'variable fz_{sp} atom "{s * a.A * k:.12g}*sin({k:.12g}*z+{a.phase:.12g}){tf}"')
        lines.append(f'variable ez_{sp} atom "{s * a.A:.12g}*cos({k:.12g}*z+{a.phase:.12g}){tf}"')
    open(a.out + ".lmp", "w").write("\n".join(lines) + "\n")
    json.dump(spec, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
