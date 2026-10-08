#!/usr/bin/env python3

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

BOND_LEN = 0.97
MAX_TURN_DEG = 110.0


def random_unit(rng, n=1):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_chain(rng, n_mono, start):

    pos = np.empty((n_mono, 3))
    pos[0] = start
    u = random_unit(rng)[0]
    pos[1] = pos[0] + BOND_LEN * u
    cos_min = math.cos(math.radians(MAX_TURN_DEG))
    for i in range(2, n_mono):
        while True:
            w = random_unit(rng)[0]
            if np.dot(w, u) >= cos_min:
                break
        pos[i] = pos[i - 1] + BOND_LEN * w
        u = w
    return pos


def place_ions(rng, n_each, box, min_sep):

    n_tot = 2 * n_each
    pts = np.empty((n_tot, 3))
    placed = 0
    tries = 0
    while placed < n_tot:
        tries += 1
        if tries > 400 * n_tot:
            raise RuntimeError("could not place ions with the requested separation")
        p = rng.uniform(0.0, box, size=3)
        if placed:
            d = p - pts[:placed]
            d -= box * np.round(d / box)
            if np.min(np.linalg.norm(d, axis=1)) < min_sep:
                continue
        pts[placed] = p
        placed += 1
    return pts[:n_each], pts[n_each:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=float, required=True, help="c_LJ = N_+/N_bead")
    ap.add_argument("--rho", type=float, default=0.85, help="initial number density (all beads)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    n_ion = M.n_pairs(a.conc)
    n_atoms = M.N_BEAD + 2 * n_ion
    box = (n_atoms / a.rho) ** (1.0 / 3.0)


    coords = np.empty((n_atoms, 3))
    mol = np.empty(n_atoms, dtype=int)
    typ = np.empty(n_atoms, dtype=int)
    k = 0
    for c in range(M.N_CHAINS):
        start = rng.uniform(0.0, box, size=3)
        ch = build_chain(rng, M.N_MONO, start)
        coords[k:k + M.N_MONO] = ch
        mol[k:k + M.N_MONO] = c + 1
        typ[k:k + M.N_MONO] = M.T_MONO
        k += M.N_MONO


    cat, ani = place_ions(rng, n_ion, box, min_sep=1.3 * M.WCA)
    coords[k:k + n_ion] = cat
    typ[k:k + n_ion] = M.T_CAT
    mol[k:k + n_ion] = np.arange(M.N_CHAINS + 1, M.N_CHAINS + 1 + n_ion)
    k += n_ion
    coords[k:k + n_ion] = ani
    typ[k:k + n_ion] = M.T_ANI
    mol[k:k + n_ion] = np.arange(M.N_CHAINS + 1 + n_ion, M.N_CHAINS + 1 + 2 * n_ion)
    k += n_ion
    assert k == n_atoms


    img = np.floor(coords / box).astype(int)
    wrapped = coords - img * box


    bonds = []
    for c in range(M.N_CHAINS):
        base = c * M.N_MONO + 1
        for i in range(M.N_MONO - 1):
            bonds.append((base + i, base + i + 1))

    with open(a.out, "w") as f:
        f.write(f"LAMMPS data: KG melt + salt, c_LJ={a.conc}, seed={a.seed}\n\n")
        f.write(f"{n_atoms} atoms\n{len(bonds)} bonds\n\n")
        f.write("3 atom types\n1 bond types\n\n")
        for d in "xyz":
            f.write(f"0.0 {box:.10f} {d}lo {d}hi\n")
        f.write("\nMasses\n\n")
        for t in (M.T_MONO, M.T_CAT, M.T_ANI):
            f.write(f"{t} {M.MASS_T[t]}\n")
        f.write("\nAtoms # full\n\n")
        for i in range(n_atoms):
            q = M.CHARGE_T[typ[i]]
            f.write(f"{i+1} {mol[i]} {typ[i]} {q:.6f} "
                    f"{wrapped[i,0]:.6f} {wrapped[i,1]:.6f} {wrapped[i,2]:.6f} "
                    f"{img[i,0]} {img[i,1]} {img[i,2]}\n")
        f.write("\nBonds\n\n")
        for i, (p, q_) in enumerate(bonds, 1):
            f.write(f"{i} 1 {p} {q_}\n")


    ee = coords[M.N_MONO - 1::M.N_MONO][:M.N_CHAINS] - coords[0:M.N_BEAD:M.N_MONO]
    print(f"atoms={n_atoms} (mono={M.N_BEAD}, cat={n_ion}, ani={n_ion})")
    print(f"box L={box:.4f} sigma, rho_init={a.rho}")
    print(f"<Ree^2>^1/2 = {np.sqrt((ee**2).sum(1).mean()):.3f} sigma "
          f"(ideal {BOND_LEN*math.sqrt(M.N_MONO-1):.3f}+)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
