#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio
import model as M
from analyze_zerofield import msd_fft, fit_D, structure_factors

MODES = (2, 3, 4, 5, 6)
SLOPE_LO, SLOPE_HI = 1000.0, 6000.0


def tau_int_complex(a, c_window=6.0):

    out = 0.0
    for comp in (a.real, a.imag):
        x = comp - comp.mean()
        n = len(x)
        c = np.correlate(x, x, "full")[n - 1:]
        c = c[: n // 4]
        if c[0] <= 0:
            continue
        rho = c / c[0]
        run = 0.5
        tau = 0.5
        for w in range(1, len(rho)):
            run += rho[w]
            tau = run
            if w >= c_window * max(run, 0.5):
                break
        out = max(out, tau)
    return max(out, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conc", type=float, default=None)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--eps-r", type=float, default=M.EPS_R,
                    help="relative permittivity of the run; sets the l_B saved to Sk.npz, "
                         "which learn/ uses for the Coulomb term")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sk-frames", type=int, default=200)
    a = ap.parse_args()

    dump = os.path.join(a.run, "ions.dump")
    steps, boxes, cols, data = dumpio.read_dump(dump, stride=a.stride)
    typ = data[0, :, cols.index("type")].astype(int)
    r = np.stack([data[:, :, cols.index(c)] for c in ("xu", "yu", "zu")], axis=2)
    del data
    t = (steps - steps[0]) * M.DT * a.stride
    lz = float(boxes[0][2, 1] - boxes[0][2, 0])
    vol = float(np.prod(boxes[0][:, 1] - boxes[0][:, 0]))
    cat, ani = typ == M.T_CAT, typ == M.T_ANI
    npair = int(cat.sum())

    print("=" * 78)
    print(f"zero-field report  {a.run}")
    print(f"  {len(t)} frames, {t[-1]:.0f} tau, {npair} ion pairs, L = {lz:.4f}, V = {vol:.1f}")


    com = r.mean(axis=1)
    vcom = float(np.linalg.norm(np.polyfit(t, com, 1)[0]))
    r = r - com[:, None, :]
    print(f"  COM drift {vcom:.2e} sigma/tau   {'OK' if vcom < 1e-3 else 'FAIL (>1e-3)'}")


    D, slopes = {}, {}
    for lab, m in (("cation", cat), ("anion", ani)):
        msd = msd_fft(r[:, m, :])
        k = (t >= SLOPE_LO) & (t <= min(SLOPE_HI, 0.5 * t[-1]))
        sl = float(np.polyfit(np.log(t[k]), np.log(msd[k]), 1)[0])
        D[lab] = float(fit_D(t[1:], msd[1:]))
        slopes[lab] = sl
        np.savetxt(os.path.join(a.run, f"msd_{lab}.dat"), np.c_[t, msd], header="t msd")
    print(f"  D_cation = {D['cation']:.4e}  (log-log slope over lags "
          f"{SLOPE_LO:.0f}-{SLOPE_HI:.0f} tau: {slopes['cation']:.3f})")
    print(f"  D_anion  = {D['anion']:.4e}  (slope {slopes['anion']:.3f})")
    diffusive = all(0.90 <= s <= 1.05 for s in slopes.values())
    print(f"  diffusive regime reached: {'yes' if diffusive else 'NO -- D is an upper bound'}")


    tau = {}
    dt_frame = t[1] - t[0]
    print(f"  tau_int of the density modes (complex, real and imaginary parts):")
    print(f"    {'m':>3} {'lambda':>7} {'cation':>10} {'anion':>10}   (tau)")
    for m in MODES:
        k = 2.0 * np.pi * m / lz
        row = {}
        for lab, sel in (("cation", cat), ("anion", ani)):
            amp = np.exp(-1j * k * r[:, sel, 2]).mean(axis=1)
            row[lab] = float(tau_int_complex(amp) * dt_frame)
        tau[m] = row
        print(f"    {m:3d} {lz/m:7.2f} {row['cation']:10.0f} {row['anion']:10.0f}")


    IN_FIELD = 2.0
    Ds = min(D["cation"], D["anion"])
    k3 = 2.0 * np.pi * 3 / lz
    t3_tracer = 1.0 / (k3 * k3 * Ds)
    t3_eff = IN_FIELD * t3_tracer
    t3_meas = max(tau[3]["cation"], tau[3]["anion"])
    ntrans = int(np.ceil(10 * t3_eff / M.DT / 2e5) * 2e5)
    nprod = int(np.ceil(50 * t3_eff / M.DT / 2e5) * 2e5)
    print(f"  m=3: 1/(k^2 D_s) = {t3_tracer:.0f} tau, measured tau_int = {t3_meas:.0f} tau "
          f"(noisy), tau_eff = {IN_FIELD}x tracer = {t3_eff:.0f} tau")
    print(f"  -> transient {ntrans:.3g} steps ({10*t3_eff:.0f} tau), "
          f"production {nprod:.3g} steps ({50*t3_eff:.0f} tau)")


    sel = np.linspace(0, len(t) - 1, min(a.sk_frames, len(t))).astype(int)
    pos = r[sel] + com[sel][:, None, :]
    signs = np.where(cat, 1.0, -1.0)
    sk = structure_factors(pos, signs, vol ** (1.0 / 3.0))
    kk, szz, snn = sk["k"], sk["ZZ"], sk["NN"]
    n_each = npair / vol
    lB = M.bjerrum_sigma(a.temp, a.eps_r)
    kD2 = 8.0 * np.pi * lB * n_each
    ratio = float((szz[0] / (2 * n_each)) / (kk[0] ** 2 / kD2))
    print(f"  S_ZZ/(2n) at k_min = {kk[0]:.3f}: ratio to k^2/kappa_D^2 = {ratio:.3f} "
          f"{'OK' if abs(ratio - 1) <= 0.05 else '(outside 1 +- 0.05)'}")
    np.savez(os.path.join(a.run, "Sk.npz"), n_each=n_each, lB=lB, kappaD2=kD2,
             ratio_kmin=ratio, **{f"S_{k}" if k not in ("k", "n2") else k: v
                                  for k, v in sk.items()})

    json.dump(dict(D_cation=D["cation"], D_anion=D["anion"], slopes=slopes,
                   diffusive=diffusive, v_com=vcom, volume=vol, lz=lz,
                   n_pairs=npair, tau_tot=float(t[-1]), conc=a.conc),
              open(os.path.join(a.run, "D.json"), "w"), indent=2)
    json.dump(dict(tau_int=tau, tau_int_m3_measured=t3_meas,
                   tau_tracer_m3=t3_tracer, in_field_factor=IN_FIELD,
                   tau_eff_m3=t3_eff, ntrans=ntrans, nprod=nprod,
                   dt_frame=float(dt_frame)),
              open(os.path.join(a.run, "tau_int.json"), "w"), indent=2)
    print(f"  wrote D.json, tau_int.json, Sk.npz")
    print("=" * 78)


if __name__ == "__main__":
    main()
