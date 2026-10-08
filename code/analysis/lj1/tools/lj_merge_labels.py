#!/usr/bin/env python3
"""One-label profiles of an LJ field run: the two identical labels of the MD (types 1 and 2, N/2 each) are merged
into the single species of the one-component fluid, directly from the LAMMPS ave/chunk blocks (prof_cat.dat +
prof_ani.dat): per block n = n_1 + n_2, n<f_z> = sum of the two force densities, likewise the applied force density;
f_int = f_tot - f_ext; means and standard errors over `--nblock` merged blocks, exactly as tools/lj_profiles.py does
per label.  Units k_B T.  Writes profiles.npz with the single species stored under the 'cation_*' keys (learn/ reads the
species present in the file) and acceptance.json with the merged-profile checks; `pass` is taken from the two-label
acceptance of the same run (same runs and splits as the two-label benchmark), the merged checks are stored beside it.
    lj_merge_labels.py --run <two-label field dir> --pot <pNN.json> --out <one-label field dir> [--nblock 8]
"""
import argparse, json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lj_profiles import read_chunks, ext_pot, ext_force


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--pot", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--nblock", type=int, default=8)
    a = ap.parse_args()
    spec = json.load(open(a.pot)); lz, kT = float(spec["lz"]), float(spec["temp"])
    C1, C2 = read_chunks(os.path.join(a.run, "prof_cat.dat")), read_chunks(os.path.join(a.run, "prof_ani.dat"))
    assert C1.shape == C2.shape and np.allclose(C1[:, :, 1], C2[:, :, 1])
    B = C1.shape[0]; assert B % a.nblock == 0, f"{B} blocks"
    z = C1[0, :, 1]; dz = lz / len(z)
    n_b = C1[:, :, 3] + C2[:, :, 3]
    ft_b = C1[:, :, 3] * C1[:, :, 4] + C2[:, :, 3] * C2[:, :, 4]
    fe_b = C1[:, :, 3] * C1[:, :, 5] + C2[:, :, 3] * C2[:, :, 5]
    g = B // a.nblock; merge = lambda x: x.reshape(a.nblock, g, -1).mean(1)
    n_b, ft_b, fe_b = merge(n_b), merge(ft_b), merge(fe_b)
    m = lambda x: (x.mean(0), x.std(0, ddof=1) / np.sqrt(a.nblock))
    (n, ne), (ft, fte), (fe, fee) = m(n_b), m(ft_b), m(fe_b)
    fint, finte = ft - fe, np.sqrt(fte ** 2 + fee ** 2)
    ana = n * ext_force(spec, z); dev = np.max(np.abs(fe - ana)) / max(np.ptp(ana), 1e-12)
    dn = (np.roll(n, -1) - np.roll(n, 1)) / (2 * dz); dne = np.sqrt(np.roll(ne, -1) ** 2 + np.roll(ne, 1) ** 2) / (2 * dz)
    pull = (kT * dn - ft) / np.sqrt((kT * dne) ** 2 + fte ** 2)
    h1, h2 = n_b[: a.nblock // 2].mean(0), n_b[a.nblock // 2:].mean(0)
    drift = np.sqrt(np.mean(((h1 - h2) / (2 * ne + 1e-300)) ** 2))
    b = 1.0 / kT
    store = dict(z=z, n=n, n_err=ne, f_tot=b * ft, f_tot_err=b * fte, f_ext=b * fe, f_int=b * fint, f_int_err=b * finte,
                 V=b * ext_pot(spec, z), dV=-b * ext_force(spec, z), ybg_pull=pull)
    # no block arrays: learn/ then uses the stored (legacy diagonal) errors, the convention of the two-label benchmark
    two = json.load(open(os.path.join(a.run, "acceptance.json")))
    acc = dict(tag=spec["tag"], family=spec["family"], labels="merged (one species)",
               cation=dict(ybg_chi2=float(np.mean(pull ** 2)), ext_dev=float(dev), half_drift=float(drift),
                           n_mean=float(n.mean()), contrast=float(n.max() / max(n.min(), 1e-12))),
               vcm_z_mean=two.get("vcm_z_mean"), n_total_max=float(n.max()),
               two_label_pass=bool(two["pass"]), two_label=dict(cation=two["cation"], anion=two["anion"]),
               merged_pass=bool(np.mean(pull ** 2) < 3.0 and dev < 0.05 and drift < 3.0 and n.max() < 1.1))
    acc["pass"] = acc["two_label_pass"]
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "profiles.npz"), **{f"cation_{k}": v for k, v in store.items()},
                        lz=lz, temp=kT, energy_unit="kT", species="one label (types 1 + 2 merged)", pot_json=json.dumps(spec))
    json.dump(acc, open(os.path.join(a.out, "acceptance.json"), "w"), indent=1)
    print(f"{spec['tag']} {spec['family']}: merged YBG chi2 {acc['cation']['ybg_chi2']:.2f} ext dev {dev:.1e} drift {drift:.2f} "
          f"n_max {n.max():.3f} contrast {acc['cation']['contrast']:.1f} -> {'PASS' if acc['pass'] else 'REVIEW'}"
          f"{'' if acc['merged_pass'] == acc['two_label_pass'] else '  (merged checks differ: ' + str(acc['merged_pass']) + ')'}")


if __name__ == "__main__":
    main()
