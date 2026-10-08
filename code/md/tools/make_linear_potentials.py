#!/usr/bin/env python3
"""Weak single-mode neutral potentials for the linear-response regime.

V_N(z) = A cos(2 pi m z / L + phase) on both ions (psi = 0), with A of a few
hundredths of k_B T, so that the number-channel response dn/nbar ~ A / Gamma
stays at the 10% level.  Written with make_potential's LAMMPS expressions so
the files match the rest of the campaign.

    make_linear_potentials.py --lz 24.345259 --outdir <state>/potentials \
        --spec p40:1:0.03 p41:1:0.05 p42:2:0.03 p43:2:0.05
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_potential import lmp_force_expr, lmp_energy_expr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lz", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--spec", nargs="+", required=True, help="tag:m:A (A in k_B T, cosine amplitude)")
    ap.add_argument("--phase0", type=float, default=0.4)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for i, s in enumerate(a.spec):
        tag, m, A = s.split(":"); m = int(m); A = float(A)
        if os.path.exists(os.path.join(a.outdir, tag + ".json")):
            sys.exit(f"{tag} exists in {a.outdir}: refusing to overwrite")
        t = [dict(m=m, A=A, k=2 * np.pi * m / a.lz, phase=a.phase0 + 0.9 * i)]
        spec = dict(tag=tag, seed=-1, kind="neutral", family="linear", lz=a.lz, temp=1.0,
                    pp_neutral_kT=2 * A, pp_charged_kT=0.0, neutral=t, charged=[],
                    note=f"weak single-mode neutral drive for the linear regime, m = {m}, A = {A} kT")
        json.dump(spec, open(os.path.join(a.outdir, tag + ".json"), "w"), indent=2)
        with open(os.path.join(a.outdir, tag + ".lmp"), "w") as f:
            f.write(f"# external potential '{tag}': {spec['note']}\n")
            f.write(f"# L_z={a.lz:.10g}  modes m: V_N [{m}]  psi []\n")
            f.write(f'variable fz_cat atom "{lmp_force_expr(t, [], +1)}"\n')
            f.write(f'variable fz_ani atom "{lmp_force_expr(t, [], -1)}"\n')
            f.write(f'variable ez_cat atom "{lmp_energy_expr(t, [], +1)}"\n')
            f.write(f'variable ez_ani atom "{lmp_energy_expr(t, [], -1)}"\n')
        z = np.linspace(0, a.lz, 5001)
        v = A * np.cos(t[0]["k"] * z + t[0]["phase"])
        print(f"[{tag}] m={m} k={t[0]['k']:.4f} A={A} kT  pp={np.ptp(v):.4f} kT  max|F|={A * t[0]['k']:.4f}")


if __name__ == "__main__":
    main()
