#!/usr/bin/env python3
"""External potentials for the size test (boxes n x longer along z).

(A) in-range wavelengths: a production potential of the small box, every mode
    m -> n m, so k = 2 pi m / L is unchanged and V(z) repeats the small-box
    potential n times.  The base terms (m, A, phase) are taken from a potential
    JSON; k is recomputed from this system's own small-box edge, so the same
    base gives the same potential in reduced coordinates at either eps_r.
(B) longer wavelengths (n >= 2): the big box's m = 1 mode, k = k_1 / n, with
    peak-to-peak 1 kT, V_N only (tag p30) or V_N plus psi (p31).

    make_bigbox_potentials.py --base-dir <small potentials> --lsmall 24.5023 --n 4 --outdir <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_potential import lmp_force_expr, lmp_energy_expr

BASE_TAGS = ["p01", "p09", "p13", "p15"]      # fourier both / mixed / gauss neutral / gauss charged


def write(spec, outdir):
    tag = spec["tag"]
    with open(os.path.join(outdir, tag + ".json"), "w") as f:
        json.dump(spec, f, indent=2)
    nt, ch = spec["neutral"], spec["charged"]
    with open(os.path.join(outdir, tag + ".lmp"), "w") as f:
        f.write(f"# size test: {spec['note']}\n")
        f.write(f"# L_z={spec['lz']:.10g}  modes m: V_N {[t['m'] for t in nt]}  psi {[t['m'] for t in ch]}\n")
        f.write(f'variable fz_cat atom "{lmp_force_expr(nt, ch, +1)}"\n')
        f.write(f'variable fz_ani atom "{lmp_force_expr(nt, ch, -1)}"\n')
        f.write(f'variable ez_cat atom "{lmp_energy_expr(nt, ch, +1)}"\n')
        f.write(f'variable ez_ani atom "{lmp_energy_expr(nt, ch, -1)}"\n')


def scaled(base, n, lsmall):
    def sc(terms):
        return [dict(m=int(t["m"]) * n, A=float(t["A"]), k=2 * np.pi * int(t["m"]) / lsmall,
                     phase=float(t["phase"])) for t in terms]
    s = dict(base)
    s.update(lz=n * lsmall, neutral=sc(base.get("neutral", [])), charged=sc(base.get("charged", [])),
             size_factor=n, base_lz=base["lz"],
             note=f"{base['tag']} of the small box repeated {n}x (m -> {n} m)")
    return s


def longwave(tag, n, lsmall, kind, pp=1.0, phase=0.3):
    L = n * lsmall
    t = [dict(m=1, A=0.5 * pp, k=2 * np.pi / L, phase=phase)]
    return dict(tag=tag, seed=-1, kind=kind, family="bigbox_long", lz=L, temp=1.0,
                pp_neutral_kT=pp, pp_charged_kT=pp if kind == "both" else 0.0,
                neutral=t, charged=[dict(t[0], phase=phase + 1.1)] if kind == "both" else [],
                size_factor=n, note=f"big-box m = 1, k = k_1/{n}, pp {pp} kT, {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--lsmall", type=float, required=True, help="small-box edge of this system")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for tag in BASE_TAGS:
        base = json.load(open(os.path.join(a.base_dir, tag + ".json")))
        write(scaled(base, a.n, a.lsmall), a.outdir)
    if a.n >= 2:
        write(longwave("p30", a.n, a.lsmall, "neutral"), a.outdir)
        write(longwave("p31", a.n, a.lsmall, "both"), a.outdir)
    # check: each scaled potential equals the base potential on the small box
    z = np.linspace(0, a.lsmall, 997)
    for tag in BASE_TAGS:
        b = json.load(open(os.path.join(a.base_dir, tag + ".json")))
        s = json.load(open(os.path.join(a.outdir, tag + ".json")))
        f = lambda terms, zz, L: sum(t["A"] * np.cos(2 * np.pi * t["m"] / L * zz + t["phase"]) for t in terms)
        for part in ("neutral", "charged"):
            vb = f(b.get(part, []), z, a.lsmall) if b.get(part) else 0 * z
            vs = f(s[part], z + a.lsmall * (a.n - 1), s["lz"]) if s[part] else 0 * z
            assert np.max(np.abs(vb - vs)) < 1e-9, (tag, part, np.max(np.abs(vb - vs)))
    print(f"wrote {len(os.listdir(a.outdir)) // 2} potentials to {a.outdir}; scaled == base on the last period")


if __name__ == "__main__":
    main()
