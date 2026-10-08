#!/usr/bin/env python3
"""Profiles and acceptance of a field run with a two-dimensional potential V_alpha(x, y).

From cat.dump / ani.dump (id type x y z fx fy fz v_fx v_fy):
  maps     n(x, y), total and applied force densities on an N x N grid (N = round(L / binw)),
           means and standard errors over `--nblock` blocks;
  modes    for |m_x|, m_y <= mmax the Fourier amplitudes  a(t) = (1/V) sum_j w_j exp(-i k.r_j)
           of the density (w = 1) and of the force densities (w = f_x, f_y, f^ext_x, f^ext_y),
           computed from the particle positions (no binning), averaged over blocks of
           `--nfb` frames; errors downstream use the integrated autocorrelation time of the blocks.
Acceptance (the planar criteria of Section 7, mode by mode):
  ext      applied force against the spec, per particle;
  YBG      k_B T i k n_k = f^tot_k on the driven modes (both components), within 3 sigma;
  halves   first against second half of the run on the driven modes, within 3 sigma;
  current  mean ion velocity along x and y within 3 sigma of zero.
Driven modes: |V_k| >= 1e-3 of the largest, inside mmax.
Writes profiles2d.npz and acceptance.json in the run directory.

    analyze_profiles2d.py --run <field_pNN> --pot <pNN.json> [--mmax 15]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio
from field_acceptance import tau_int


def series(s, x):
    return sum(t["A"] * np.cos(t["k"] * x + t["phase"]) for t in s) + 0.0 * x


def dseries(s, x):
    return sum(-t["A"] * t["k"] * np.sin(t["k"] * x + t["phase"]) for t in s) + 0.0 * x


def pot(spec, sign, x, y, what="V"):
    """V, or the forces fx = -dV/dx, fy = -dV/dy, of the species with charge sign at (x, y)."""
    out = 0.0
    for t in spec["terms"]:
        a = t["amp"] * (1.0 if t["channel"] == "N" else sign)
        if what == "V":
            out = out + a * series(t["X"], x) * series(t["Y"], y)
        elif what == "fx":
            out = out - a * dseries(t["X"], x) * series(t["Y"], y)
        else:
            out = out - a * series(t["X"], x) * dseries(t["Y"], y)
    return out


def err_ac(x):
    """Standard error of the mean of a (complex) block series, each part with its own tau_int."""
    f = lambda v: v.var(ddof=1) * 2 * tau_int(v) / len(v)
    return np.sqrt(f(x.real) + (f(x.imag) if np.iscomplexobj(x) else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pot", required=True)
    ap.add_argument("--binw", type=float, default=0.1)
    ap.add_argument("--nblock", type=int, default=8)
    ap.add_argument("--mmax", type=int, default=15)
    ap.add_argument("--nfb", type=int, default=100, help="frames per block of the mode series")
    a = ap.parse_args()
    spec = json.load(open(a.pot))
    L, kT = float(spec["lz"]), float(spec["temp"])
    N = int(round(L / a.binw)); dx = L / N
    xc = (np.arange(N) + 0.5) * dx
    edges = np.linspace(0.0, L, N + 1)
    k1 = 2 * np.pi / L
    mx = np.arange(-a.mmax, a.mmax + 1); my = np.arange(0, a.mmax + 1)
    V = L ** 3
    # driven modes from the potential itself
    Xg, Yg = np.meshgrid(xc, xc, indexing="ij")
    drv = np.zeros((len(mx), len(my)), bool)
    for sign in (1.0, -1.0):
        Vk = np.fft.fft2(pot(spec, sign, Xg, Yg)) / N ** 2
        for i, m1 in enumerate(mx):
            for j, m2 in enumerate(my):
                if (m1, m2) != (0, 0) and not (m2 == 0 and m1 < 0) and abs(Vk[m1 % N, m2 % N]) >= 1e-3 * np.abs(Vk).max():
                    drv[i, j] = True
    store = dict(lz=L, temp=kT, x=xc, mode_mx=mx, mode_my=my, driven=drv, pot_json=json.dumps(spec))
    acc = dict(tag=spec["tag"], family=spec["family"], n_driven=int(drv.sum()))
    print("=" * 74)
    print(f"2D field run {a.run}: {spec['tag']} {spec['family']} {spec['kind']}, {N} x {N} bins, {int(drv.sum())} driven modes (|m| <= {a.mmax})")
    for lab, fn, sign in (("cation", "cat.dump", 1.0), ("anion", "ani.dump", -1.0)):
        steps, boxes, cols, data = dumpio.read_dump(os.path.join(a.run, fn))
        lo = boxes[0][:, 0]
        if abs((boxes[0][0, 1] - boxes[0][0, 0]) - L) > 1e-4:
            raise ValueError(f"box {boxes[0][0]} does not match L = {L}")
        ix, iy = cols.index("x"), cols.index("y")
        x = (data[:, :, ix] - lo[0]) % L; y = (data[:, :, iy] - lo[1]) % L
        w = dict(ftx=data[:, :, cols.index("fx")], fty=data[:, :, cols.index("fy")],
                 fex=data[:, :, cols.index([c for c in cols if c.startswith("v_fx")][0])],
                 fey=data[:, :, cols.index([c for c in cols if c.startswith("v_fy")][0])])
        F = len(steps)
        # the box coordinates the LAMMPS variables saw
        xa, ya = data[:, :, ix], data[:, :, iy]
        sel = np.linspace(0, F - 1, min(20, F)).astype(int)
        rng = max(np.ptp(pot(spec, sign, Xg + lo[0], Yg + lo[1], "fx")), 1e-12)
        dev = max(np.abs(w["fex"][sel] - pot(spec, sign, xa[sel], ya[sel], "fx")).max(),
                  np.abs(w["fey"][sel] - pot(spec, sign, xa[sel], ya[sel], "fy")).max()) / rng
        # ---- maps in blocks
        vbin = dx * dx * L
        maps = {k: [] for k in ("n", "ftx", "fty", "fex", "fey")}
        for bl in np.array_split(np.arange(F), a.nblock):
            xb, yb = x[bl].ravel(), y[bl].ravel()
            maps["n"].append(np.histogram2d(xb, yb, bins=(edges, edges))[0] / (len(bl) * vbin))
            for k in ("ftx", "fty", "fex", "fey"):
                maps[k].append(np.histogram2d(xb, yb, bins=(edges, edges), weights=w[k][bl].ravel())[0] / (len(bl) * vbin))
        m = {k: (np.mean(v, 0), np.std(v, 0, ddof=1) / np.sqrt(a.nblock)) for k, v in maps.items()}
        # ---- mode series from the particles
        nb = F // a.nfb
        ser = {k: np.zeros((nb, len(mx), len(my)), complex) for k in ("n", "ftx", "fty", "fex", "fey")}
        for b in range(nb):
            fr = slice(b * a.nfb, (b + 1) * a.nfb)
            Ex = np.exp(-1j * k1 * x[fr][:, :, None] * mx[None, None, :])
            Ey = np.exp(-1j * k1 * y[fr][:, :, None] * my[None, None, :])
            ser["n"][b] = np.einsum("fnx,fny->xy", Ex, Ey) / (a.nfb * V)
            for k in ("ftx", "fty", "fex", "fey"):
                ser[k][b] = np.einsum("fnx,fny->xy", Ex * w[k][fr][:, :, None], Ey) / (a.nfb * V)
        # ---- acceptance on the driven modes
        kx = (k1 * mx)[:, None] * np.ones(len(my))[None, :]; ky = np.ones(len(mx))[:, None] * (k1 * my)[None, :]
        ybg, half = [], []
        h = nb // 2
        for i, j in zip(*np.where(drv)):
            for kc, fk in ((kx[i, j], "ftx"), (ky[i, j], "fty")):
                if kc == 0.0:
                    continue
                d = kT * 1j * kc * ser["n"][:, i, j] - ser[fk][:, i, j]
                ybg.append(abs(d.mean()) / max(err_ac(d), 1e-300))
            s = ser["n"][:, i, j]
            half.append(abs(s[:h].mean() - s[h:].mean()) / np.sqrt(err_ac(s[:h]) ** 2 + err_ac(s[h:]) ** 2))
        acc[lab] = dict(ext_dev=float(dev), ybg_max=float(np.max(ybg)), ybg_median=float(np.median(ybg)),
                        half_max=float(np.max(half)), half_median=float(np.median(half)),
                        n_mean=float(m["n"][0].mean()), contrast=float(m["n"][0].max() / max(m["n"][0].min(), 1e-12)),
                        frames=int(F), blocks=int(nb))
        print(f"  {lab}: {F} frames; applied force vs spec {dev:.1e}; YBG driven modes max {np.max(ybg):.2f} (median {np.median(ybg):.2f}) sigma; "
              f"halves max {np.max(half):.2f} (median {np.median(half):.2f}) sigma; contrast {acc[lab]['contrast']:.1f}")
        store.update({f"{lab}_n": m["n"][0], f"{lab}_n_err": m["n"][1],
                      f"{lab}_fx_int": m["ftx"][0] - m["fex"][0], f"{lab}_fy_int": m["fty"][0] - m["fey"][0],
                      f"{lab}_fx_int_err": np.sqrt(m["ftx"][1] ** 2 + m["fex"][1] ** 2),
                      f"{lab}_fy_int_err": np.sqrt(m["fty"][1] ** 2 + m["fey"][1] ** 2),
                      f"{lab}_V": pot(spec, sign, Xg + lo[0], Yg + lo[1]),
                      **{f"{lab}_mode_{k}": v for k, v in ser.items()}})
        store["lo"] = lo
        del data, x, y, w
    cur = np.vstack([np.loadtxt(f, comments="#", ndmin=2) for f in sorted(glob.glob(os.path.join(a.run, "current*.dat")))])
    cur = cur[np.argsort(cur[:, 0])]
    pulls = []
    for i, nm in ((1, "vx_cation"), (2, "vx_anion"), (3, "vy_cation"), (4, "vy_anion")):
        v = cur[:, i]; e = v.std(ddof=1) * np.sqrt(2 * tau_int(v) / len(v))
        acc[nm] = [float(v.mean()), float(e)]; pulls.append(abs(v.mean()) / e)
    acc["current_max"] = float(max(pulls))
    acc["pass"] = bool(all(acc[l]["ext_dev"] < 1e-4 and acc[l]["ybg_max"] < 3 and acc[l]["half_max"] < 3 for l in ("cation", "anion"))
                       and acc["current_max"] < 3)
    print(f"  mean ion velocity: max {acc['current_max']:.2f} sigma from zero")
    print(f"  -> {'PASS' if acc['pass'] else 'REVIEW'}")
    np.savez_compressed(os.path.join(a.run, "profiles2d.npz"), **store)
    json.dump(acc, open(os.path.join(a.run, "acceptance.json"), "w"), indent=1)
    print("=" * 74)


if __name__ == "__main__":
    main()
