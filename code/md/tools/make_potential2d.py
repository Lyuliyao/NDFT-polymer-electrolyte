#!/usr/bin/env python3
"""Two-dimensional static external potentials V_alpha(x, y) = V_N(x, y) + s_alpha psi(x, y), uniform along z.

Each part is a sum of product terms  amp * X(x) * Y(y)  with X, Y cosine series
sum_i A_i cos(k_i x + phase_i) (k = 0 for a constant), k_i = 2 pi m_i / L, so every
potential is exactly box periodic.  Four potentials per state point, written as JSON
(for the analysis) and as a LAMMPS include with the atom-style variables
fx_cat fy_cat e_cat fx_ani fy_ani e_ani:

  p50  egg carton: V_N = A [cos(k_4 x) + cos(k_4 y)], psi = B [cos(k_3 x) + cos(k_5 y)]    (separable sums)
  p51  neutral rods: Gaussian wells on a square lattice of period L/3, Gaussian barriers
       at the cell corners (the planar family 'gauss' in two dimensions)
  p52  charged rods: psi = Gaussian wells on the same lattice (cation wells = anion barriers)
  p53  checkerboard: V_N = A cos(k_3 x) cos(k_4 y), psi = B cos(k_4 x) cos(k_3 y)          (oblique wavevectors (3, +-4))

    make_potential2d.py --lz 24.416564 --outdir <state>/potentials
"""
import argparse
import json
import os
import sys

import numpy as np

TOL = 1e-10


def train(lz, q, width, centre):
    """Periodic train of unit Gaussians of period lz/q as a cosine series (with its constant)."""
    P = lz / q
    out = [dict(m=0, A=float(width * np.sqrt(2 * np.pi) / P), k=0.0, phase=0.0)]
    for n in range(1, 400):
        k = 2 * np.pi * n / P
        c = 2 * width * np.sqrt(2 * np.pi) / P * np.exp(-0.5 * (k * width) ** 2)
        if c < TOL:
            break
        out.append(dict(m=int(n * q), A=float(c), k=float(k), phase=float(-k * centre)))
    return out


def cosine(lz, m, phase):
    return [dict(m=int(m), A=1.0, k=float(2 * np.pi * m / lz), phase=float(phase))]


ONE = [dict(m=0, A=1.0, k=0.0, phase=0.0)]


def series(s, x):
    return sum(t["A"] * np.cos(t["k"] * x + t["phase"]) for t in s)


def dseries(s, x):
    return sum(-t["A"] * t["k"] * np.sin(t["k"] * x + t["phase"]) for t in s)


def part(terms, channel, X, Y):
    """V of one channel ('N' or 'Z') on the grids X, Y."""
    return sum(t["amp"] * series(t["X"], X) * series(t["Y"], Y) for t in terms if t["channel"] == channel) + 0.0 * X


def lmp_series(s, var, deriv=False):
    out = ""
    for t in s:
        if deriv:
            if t["k"] == 0.0:
                continue
            out += f"{-t['A'] * t['k']:+.12g}*sin({t['k']:.12g}*{var}{t['phase']:+.12g})"
        else:
            out += f"{t['A']:+.12g}" if t["k"] == 0.0 else f"{t['A']:+.12g}*cos({t['k']:.12g}*{var}{t['phase']:+.12g})"
    return "(" + (out.lstrip("+") or "0.0") + ")"


def lmp_expr(terms, sign, what):
    """what: 'e' (energy), 'fx', 'fy' (forces = -dV/dx, -dV/dy) for the species with charge sign."""
    out = ""
    for t in terms:
        a = t["amp"] * (1.0 if t["channel"] == "N" else sign)
        if what == "e":
            out += f"{a:+.12g}*{lmp_series(t['X'], 'x')}*{lmp_series(t['Y'], 'y')}"
        elif what == "fx":
            out += f"{-a:+.12g}*{lmp_series(t['X'], 'x', True)}*{lmp_series(t['Y'], 'y')}"
        else:
            out += f"{-a:+.12g}*{lmp_series(t['X'], 'x')}*{lmp_series(t['Y'], 'y', True)}"
    return out.lstrip("+") or "0.0"


def build(lz):
    rng = np.random.default_rng(20261002)
    ph = lambda: float(rng.uniform(0, 2 * np.pi))
    P = lz / 3.0
    x0, y0 = float(rng.uniform(0, P)), float(rng.uniform(0, P))
    pots = {}
    pots["p50"] = ("2d_egg", "both", [
        dict(channel="N", amp=0.625, X=cosine(lz, 4, ph()), Y=ONE), dict(channel="N", amp=0.625, X=ONE, Y=cosine(lz, 4, ph())),
        dict(channel="Z", amp=0.375, X=cosine(lz, 3, ph()), Y=ONE), dict(channel="Z", amp=0.375, X=ONE, Y=cosine(lz, 5, ph()))],
        "separable sums: V_N = A[cos(k4 x) + cos(k4 y)], psi = B[cos(k3 x) + cos(k5 y)]")
    pots["p51"] = ("2d_rods", "neutral", [
        dict(channel="N", amp=-2.5, X=train(lz, 3, 0.8, x0), Y=train(lz, 3, 0.8, y0)),
        dict(channel="N", amp=2.0, X=train(lz, 3, 1.0, x0 + 0.5 * P), Y=train(lz, 3, 1.0, y0 + 0.5 * P))],
        "Gaussian rods on a square lattice of period L/3: wells -2.5 kT (width 0.8), barriers +2 kT (width 1) at the corners")
    pots["p52"] = ("2d_rods", "charged", [
        dict(channel="Z", amp=-2.0, X=train(lz, 3, 0.8, x0), Y=train(lz, 3, 0.8, y0))],
        "charged Gaussian rods, period L/3: psi wells -2 kT (width 0.8): cation wells, anion barriers")
    pots["p53"] = ("2d_checker", "both", [
        dict(channel="N", amp=1.5, X=cosine(lz, 3, ph()), Y=cosine(lz, 4, ph())),
        dict(channel="Z", amp=0.5, X=cosine(lz, 4, ph()), Y=cosine(lz, 3, ph()))],
        "products: V_N = A cos(k3 x) cos(k4 y), psi = B cos(k4 x) cos(k3 y); wavevectors (3, +-4) and (4, +-3)")
    return pots, dict(period=P, centre=[x0, y0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lz", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--temp", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    pots, lattice = build(a.lz)
    x = np.linspace(0, a.lz, 601); X, Y = np.meshgrid(x, x, indexing="ij")
    for tag, (fam, kind, terms, note) in pots.items():
        j = os.path.join(a.outdir, tag + ".json")
        if os.path.exists(j):
            sys.exit(f"{j} exists: refusing to overwrite")
        vn, ps = part(terms, "N", X, Y), part(terms, "Z", X, Y)
        spec = dict(tag=tag, seed=20261002, kind=kind, family=fam, dim=2, lz=a.lz, temp=a.temp,
                    pp_neutral_kT=float(np.ptp(vn)), pp_charged_kT=float(np.ptp(ps)), terms=terms, note=note)
        if fam == "2d_rods":
            spec["lattice"] = lattice
        json.dump(spec, open(j, "w"), indent=1)
        with open(os.path.join(a.outdir, tag + ".lmp"), "w") as f:
            f.write(f"# two-dimensional external potential '{tag}' ({fam}, {kind}): {note}\n# L = {a.lz:.10g}\n")
            for sp, sign in (("cat", 1.0), ("ani", -1.0)):
                for what in ("fx", "fy", "e"):
                    f.write(f'variable {what}_{sp} atom "{lmp_expr(terms, sign, what)}"\n')
        fmax = max(np.abs(sum(t["amp"] * (1 if t["channel"] == "N" else s) * dseries(t["X"], X) * series(t["Y"], Y) for t in terms)).max()
                   for s in (1.0, -1.0))
        print(f"[{tag}] {fam:10s} {kind:8s} pp(V_N) {np.ptp(vn):.2f}  pp(psi) {np.ptp(ps):.2f}  V_+ [{(vn + ps).min():+.2f}, {(vn + ps).max():+.2f}]  "
              f"V_- [{(vn - ps).min():+.2f}, {(vn - ps).max():+.2f}] kT  max|F_x| {fmax:.2f}  line {max(len(lmp_expr(terms, 1.0, w)) for w in ('fx', 'fy', 'e'))} chars")


if __name__ == "__main__":
    main()
