#!/usr/bin/env python3
"""Profiles of one LJ field run from the LAMMPS ave/chunk files (no dumps).

Per block: number density n, total force density n<f_z> and applied external
force density n<f_z^ext> of each label; f_int = f_tot - f_ext.  Means and
standard errors over `--nblock` blocks (consecutive LAMMPS blocks are merged),
all energies and forces stored in units of k_B T (V/kT, f/kT: learn/ works with
k_B T = 1, the ion campaign's T),
with the error of f_int taken as sqrt(err_tot^2 + err_ext^2), the convention of
tools/analyze_profiles.py.  Writes profiles.npz with the keys of that script
(label 1 stored as 'cation', label 2 as 'anion') and acceptance.json.

    lj_profiles.py --run <field dir> --pot <pNN.json> [--nblock 8]
"""
import argparse
import json
import os

import numpy as np


def read_chunks(path):
    """(B, nchunk, 6): Chunk Coord1 Ncount density/number fz v_fz per block."""
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    out, i = [], 0
    while i < len(rows):
        nch = int(rows[i][1])
        out.append(np.array(rows[i + 1:i + 1 + nch], float))
        i += 1 + nch
    return np.array(out)


def ext_pot(spec, z):
    return sum(t["A"] * np.cos(t["k"] * z + t["phase"]) for t in spec["neutral"]) + 0.0 * z


def ext_force(spec, z):
    return sum(t["A"] * t["k"] * np.sin(t["k"] * z + t["phase"]) for t in spec["neutral"]) + 0.0 * z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pot", required=True)
    ap.add_argument("--nblock", type=int, default=8)
    a = ap.parse_args()
    spec = json.load(open(a.pot))
    if spec["charged"]:
        raise ValueError("LJ runs use neutral potentials only")
    lz, kT = float(spec["lz"]), float(spec["temp"])
    store, acc = {}, {"tag": spec["tag"], "family": spec["family"]}
    ntot = None
    for lab, fn in (("cation", "prof_cat.dat"), ("anion", "prof_ani.dat")):
        C = read_chunks(os.path.join(a.run, fn))
        B = C.shape[0]
        if B % a.nblock:
            raise ValueError(f"{B} LAMMPS blocks cannot be merged into {a.nblock}")
        z = C[0, :, 1]
        dz = lz / len(z)
        if abs(z[0] - 0.5 * dz) > 1e-6 or abs(z[1] - z[0] - dz) > 1e-6:
            raise ValueError(f"bins {z[:2]} are not the centres of {len(z)} bins on [0, {lz}]")
        n_b = C[:, :, 3]
        ft_b = n_b * C[:, :, 4]
        fe_b = n_b * C[:, :, 5]
        g = B // a.nblock
        merge = lambda x: x.reshape(a.nblock, g, -1).mean(1)
        n_b, ft_b, fe_b = merge(n_b), merge(ft_b), merge(fe_b)
        m = lambda x: (x.mean(0), x.std(0, ddof=1) / np.sqrt(a.nblock))
        (n, ne), (ft, fte), (fe, fee) = m(n_b), m(ft_b), m(fe_b)
        fint, finte = ft - fe, np.sqrt(fte ** 2 + fee ** 2)
        # applied force against the spec, bin averages: differs from the value at the
        # centre by O(dz^2 V''') only
        ana = n * ext_force(spec, z)
        dev = np.max(np.abs(fe - ana)) / max(np.ptp(ana), 1e-12)
        # first YBG equation, kT d_z n = f_tot, bin by bin
        dn = (np.roll(n, -1) - np.roll(n, 1)) / (2 * dz)
        dne = np.sqrt(np.roll(ne, -1) ** 2 + np.roll(ne, 1) ** 2) / (2 * dz)
        pull = (kT * dn - ft) / np.sqrt((kT * dne) ** 2 + fte ** 2)
        # first half against second half of the run (drift, e.g. slow freezing): with
        # independent blocks var(h1 - h2) = 4 err^2, so the rms pull is ~1 (1.2 for t(7))
        h1, h2 = n_b[: a.nblock // 2].mean(0), n_b[a.nblock // 2:].mean(0)
        drift = np.sqrt(np.mean(((h1 - h2) / (2 * ne + 1e-300)) ** 2))
        b = 1.0 / kT                                   # energies and forces in units of k_B T
        store[lab] = dict(z=z, n=n, n_err=ne, f_tot=b * ft, f_tot_err=b * fte, f_ext=b * fe, f_int=b * fint,
                          f_int_err=b * finte, V=b * ext_pot(spec, z), dV=-b * ext_force(spec, z), ybg_pull=pull)
        acc[lab] = dict(ybg_chi2=float(np.mean(pull ** 2)), ext_dev=float(dev), half_drift=float(drift),
                        n_mean=float(n.mean()), contrast=float(n.max() / max(n.min(), 1e-12)))
        ntot = n if ntot is None else ntot + n
    cur = np.loadtxt(os.path.join(a.run, "current.dat"))
    acc["vcm_z_mean"] = [float(cur[:, 1].mean()), float(cur[:, 2].mean())]
    acc["n_total_max"] = float(ntot.max())
    acc["pass"] = bool(all(acc[l]["ybg_chi2"] < 3.0 and acc[l]["ext_dev"] < 0.05 and acc[l]["half_drift"] < 3.0
                           for l in ("cation", "anion")) and acc["n_total_max"] < 1.1)
    np.savez_compressed(os.path.join(a.run, "profiles.npz"),
                        **{f"{k}_{kk}": vv for k, v in store.items() for kk, vv in v.items()},
                        lz=lz, temp=kT, energy_unit="kT", pot_json=json.dumps(spec))
    json.dump(acc, open(os.path.join(a.run, "acceptance.json"), "w"), indent=1)
    print(f"{spec['tag']} {spec['family']}: YBG chi2 {acc['cation']['ybg_chi2']:.2f}/{acc['anion']['ybg_chi2']:.2f}  "
          f"ext dev {acc['cation']['ext_dev']:.1e}  half drift {acc['cation']['half_drift']:.2f}/{acc['anion']['half_drift']:.2f}  "
          f"n_max {acc['n_total_max']:.3f}  contrast {acc['cation']['contrast']:.1f}  -> {'PASS' if acc['pass'] else 'REVIEW'}")


if __name__ == "__main__":
    main()
