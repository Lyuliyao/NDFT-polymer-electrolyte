# Copy of tools/analyze_profiles.py as of 2026-10-03, before the paired-block error model (commit 9e21a8d):
# used for the grid-refinement test, so that the finer profiles carry the same error model as the paper models.
#!/usr/bin/env python3
"""Bin the external-field runs and test the first Yvon-Born-Green equation.

    k_B T d_z n_a(z) = f_a^int(z) - n_a(z) d_z V_a(z)

Both sides are measured.  Because the external force density is exactly
f_a^ext(z) = -n_a(z) V_a'(z), the identity is equivalent to

    k_B T d_z n_a(z) = f_a^tot(z),

with f^tot the density of the total force LAMMPS reports in the dump.  The
check is done twice: bin by bin with block-average error bars, and mode by
mode in Fourier space, which is where the signal actually lives (the applied
potential contains only m in {3,4,5,6}).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import dumpio


def ext_force(spec, sign, z):
    """F_z = -dV_a/dz from the stored potential spec."""
    out = np.zeros_like(z)
    for t in spec["neutral"]:
        out += t["A"] * t["k"] * np.sin(t["k"] * z + t["phase"])
    for t in spec["charged"]:
        out += sign * t["A"] * t["k"] * np.sin(t["k"] * z + t["phase"])
    return out


def ext_pot(spec, sign, z):
    out = np.zeros_like(z)
    for t in spec["neutral"]:
        out += t["A"] * np.cos(t["k"] * z + t["phase"])
    for t in spec["charged"]:
        out += sign * t["A"] * np.cos(t["k"] * z + t["phase"])
    return out


def bin_species(dumpfile, lz, binw, nblock, zlo, spec=None, sign=None):
    """Return per-block (n(z), f_tot(z), f_ext(z)), the bin centres, and the
    per-particle deviation between the external force LAMMPS applied and the
    stored spec.  That check is done particle by particle, not bin by bin: a
    bin average compared with the analytic value at the bin centre differs by
    O(binw^2 * V'), which is a property of the binning, not an error."""
    steps, boxes, cols, data = dumpio.read_dump(dumpfile)
    iz, ifz = cols.index("z"), cols.index("fz")
    iex = cols.index([c for c in cols if c.startswith("v_fz")][0])
    pp_dev = None
    if spec is not None:
        sel = np.linspace(0, len(steps) - 1, min(20, len(steps))).astype(int)
        zz = data[sel][:, :, iz].ravel()
        ff = data[sel][:, :, iex].ravel()
        ana = ext_force(spec, sign, zz)
        pp_dev = np.max(np.abs(ff - ana)) / max(np.ptp(ana), 1e-12)
    nbin = int(round(lz / binw))
    edges = np.linspace(0.0, lz, nbin + 1)
    zc = 0.5 * (edges[1:] + edges[:-1])
    vbin = (boxes[0][0, 1] - boxes[0][0, 0]) * (boxes[0][1, 1] - boxes[0][1, 0]) * (lz / nbin)

    nfr = len(steps)
    bl = np.array_split(np.arange(nfr), nblock)
    n_b, ft_b, fe_b = [], [], []
    for sel in bl:
        z = (data[sel][:, :, iz] - zlo) % lz
        ft = data[sel][:, :, ifz]
        fe = data[sel][:, :, iex]
        cnt, _ = np.histogram(z.ravel(), bins=edges)
        sft, _ = np.histogram(z.ravel(), bins=edges, weights=ft.ravel())
        sfe, _ = np.histogram(z.ravel(), bins=edges, weights=fe.ravel())
        nf = len(sel)
        n_b.append(cnt / (nf * vbin))
        ft_b.append(sft / (nf * vbin))
        fe_b.append(sfe / (nf * vbin))
    return zc, np.array(n_b), np.array(ft_b), np.array(fe_b), pp_dev


def fourier(y, lz, mmax):
    """Complex Fourier amplitudes a_m of y(z) = sum_m a_m exp(i k_m z)."""
    n = len(y)
    f = np.fft.rfft(y) / n
    return f[:mmax + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pot", required=True, help="potential JSON")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--binw", type=float, default=0.1)
    ap.add_argument("--nblock", type=int, default=8)
    ap.add_argument("--mmax", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = json.load(open(a.pot))
    lz = spec["lz"]
    kT = a.temp
    print("=" * 74)
    print(f"YBG check  run={a.run}  potential={spec['tag']} kind={spec['kind']} "
          f"pp(V_N)={spec['pp_neutral_kT']:.2f} pp(psi)={spec['pp_charged_kT']:.2f} kT")

    store = {"lz": lz, "temp": kT, "pot": spec}
    ok_all = True
    for lab, fn, sign in (("cation", "cat.dump", +1.0), ("anion", "ani.dump", -1.0)):
        path = os.path.join(a.run, fn)
        if not os.path.exists(path) and os.path.exists(path + ".gz"):
            path += ".gz"
        # z origin of the box as written in the dump
        with dumpio._open(path) as fh:
            for _ in range(5):
                fh.readline()
            zlo = None
            l1 = fh.readline(); l2 = fh.readline(); l3 = fh.readline()
            zlo = float(l3.split()[0])
        zc, nb, ftb, feb, pp_dev = bin_species(path, lz, a.binw, a.nblock, zlo,
                                               spec=spec, sign=sign)

        n = nb.mean(0);   n_e = nb.std(0, ddof=1) / np.sqrt(a.nblock)
        ft = ftb.mean(0); ft_e = ftb.std(0, ddof=1) / np.sqrt(a.nblock)
        fe = feb.mean(0); fe_e = feb.std(0, ddof=1) / np.sqrt(a.nblock)
        fint = ft - fe
        fint_e = np.sqrt(ft_e ** 2 + fe_e ** 2)

        # --- consistency of the applied external force with the stored spec ---
        # per particle, so the result does not depend on the bin width
        print(f"\n  {lab}: applied external force vs. stored spec, per particle: "
              f"max dev {pp_dev:.2e} of range  "
              f"{'OK' if pp_dev < 1e-4 else 'MISMATCH'}")
        ok_all &= pp_dev < 1e-4

        # --- YBG, bin by bin ---------------------------------------------------
        dz = zc[1] - zc[0]
        dn = (np.roll(n, -1) - np.roll(n, 1)) / (2 * dz)
        dn_e = np.sqrt(np.roll(n_e, -1) ** 2 + np.roll(n_e, 1) ** 2) / (2 * dz)
        lhs, rhs = kT * dn, ft
        sig = np.sqrt((kT * dn_e) ** 2 + ft_e ** 2)
        pull = (lhs - rhs) / np.where(sig > 0, sig, np.inf)
        frac = np.mean(np.abs(pull) < 2.0)
        chi2 = np.mean(pull ** 2)
        print(f"        bin-by-bin: {frac:.1%} of {len(zc)} bins within 2 sigma, "
              f"chi^2/bin = {chi2:.2f}")

        # --- YBG, mode by mode -------------------------------------------------
        km = 2 * np.pi * np.arange(a.mmax + 1) / lz
        an = fourier(n, lz, a.mmax)
        af = fourier(ft, lz, a.mmax)
        # per-mode error bars from the same blocks
        anb = np.array([fourier(x, lz, a.mmax) for x in nb])
        afb = np.array([fourier(x, lz, a.mmax) for x in ftb])
        an_e = anb.std(0, ddof=1) / np.sqrt(a.nblock)
        af_e = afb.std(0, ddof=1) / np.sqrt(a.nblock)
        driven = [t["m"] for t in spec["neutral"] + spec["charged"]]
        ybg_l = kT * 1j * km * an
        ybg_r = af
        ybg_le = kT * km * np.abs(an_e)
        ybg_re = np.abs(af_e)
        pull_m = np.abs(ybg_l - ybg_r) / np.sqrt(ybg_le ** 2 + ybg_re ** 2 + 1e-300)
        print("         m  driven   kT k_m |n_m|        |f_m|         "
              "|diff|/err")
        for m in range(1, min(8, a.mmax) + 1):
            print(f"        {m:2d}  {'yes' if m in driven else ' - ':6s} "
                  f"{abs(ybg_l[m]):.5e}   {abs(ybg_r[m]):.5e}   {pull_m[m]:7.2f}")
        dm = [m for m in driven if m <= a.mmax]
        if dm:
            print(f"        driven modes only: max |diff|/err = "
                  f"{max(pull_m[m] for m in dm):.2f}")

        store[lab] = dict(z=zc, n=n, n_err=n_e, f_tot=ft, f_tot_err=ft_e,
                          f_ext=fe, f_int=fint, f_int_err=fint_e,
                          V=ext_pot(spec, sign, zc), dV=-ext_force(spec, sign, zc),
                          ybg_lhs=lhs, ybg_rhs=rhs, ybg_pull=pull,
                          m=np.arange(a.mmax + 1), km=km,
                          mode_lhs=np.abs(ybg_l), mode_rhs=np.abs(ybg_r),
                          mode_lhs_err=ybg_le, mode_rhs_err=ybg_re,
                          mode_pull=pull_m,
                          mode_driven=np.isin(np.arange(a.mmax + 1), driven))
        print(f"        n_mean = {n.mean():.5f}, contrast n_max/n_min = "
              f"{n.max()/max(n.min(),1e-12):.2f}")

    out = a.out or os.path.join(a.run, "profiles.npz")
    np.savez_compressed(out, **{f"{k}_{kk}": vv for k, v in store.items()
                                if isinstance(v, dict) for kk, vv in v.items()},
                        lz=lz, temp=kT, pot_json=json.dumps(spec))
    print(f"\nwrote {out}")
    print("=" * 74)


if __name__ == "__main__":
    main()
