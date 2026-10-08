#!/usr/bin/env python3
"""Zero-field acceptance checks (Section 7 of the plan).

  * equilibrium density from the NPT volume trace
  * ion diffusion coefficients D_+ and D_- (multi-origin MSD, FFT algorithm)
  * conductivity / Nernst-Einstein ratio
  * charge structure factor S_ZZ(k) against the Debye limit
  * the relaxation-time table of Section 6, evaluated at the measured D_s
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import dumpio
import paper_reference as PR


# ----------------------------------------------------------------- MSD (FCA)
def _autocorr(x):
    """FFT autocorrelation along axis 0 of x[T, N]."""
    T = x.shape[0]
    n = 2 * T
    f = np.fft.rfft(x, n=n, axis=0)
    ac = np.fft.irfft(f * np.conjugate(f), n=n, axis=0)[:T]
    return ac / (T - np.arange(T))[:, None]


def msd_fft(r):
    """Multi-origin MSD of r[T, N, 3], averaged over particles."""
    T, N = r.shape[0], r.shape[1]
    D = np.square(r).sum(axis=2)
    D = np.vstack([D, np.zeros((1, N))])
    S2 = sum(_autocorr(r[:, :, i]) for i in range(3))
    Q = 2.0 * D.sum(axis=0)
    S1 = np.zeros((T, N))
    for m in range(T):
        Q = Q - D[m - 1] - D[T - m]
        S1[m] = Q / (T - m)
    return (S1 - 2.0 * S2).mean(axis=1)


def fit_D(t, msd, lo=0.2, hi=0.8, dim=3):
    """Fit msd = 2*dim*D*t over the window [lo,hi] of the trace (log-spaced)."""
    i0, i1 = int(lo * len(t)), int(hi * len(t))
    i0 = max(i0, 1)
    sl, _ = np.polyfit(t[i0:i1], msd[i0:i1], 1)
    return sl / (2.0 * dim)


def msd_loglog_slope(t, msd, lo=0.2, hi=0.8):
    i0, i1 = max(int(lo * len(t)), 1), int(hi * len(t))
    return np.polyfit(np.log(t[i0:i1]), np.log(msd[i0:i1]), 1)[0]


# ------------------------------------------------------- structure factors
def structure_factors(pos, signs, L, nmax=8):
    """S_ab(k) = <rho_a(k) rho_b(-k)>/V, averaged over frames and k directions."""
    ks, idx = [], []
    for nx in range(0, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and (ny < 0 or (ny == 0 and nz <= 0)):
                    continue
                n2 = nx * nx + ny * ny + nz * nz
                if 0 < n2 <= nmax * nmax:
                    ks.append((nx, ny, nz)); idx.append(n2)
    ks = np.array(ks, dtype=float)
    idx = np.array(idx)
    kvec = 2.0 * np.pi * ks / L
    V = L ** 3

    plus = signs > 0
    acc = {k: np.zeros(len(ks)) for k in ("ZZ", "NN", "pp", "mm", "pm")}
    nfr = pos.shape[0]
    for f in range(nfr):
        ph = pos[f] @ kvec.T                       # (natom, nk)
        e = np.exp(-1j * ph)
        rp = e[plus].sum(axis=0)
        rm = e[~plus].sum(axis=0)
        acc["ZZ"] += np.abs(rp - rm) ** 2
        acc["NN"] += np.abs(rp + rm) ** 2
        acc["pp"] += np.abs(rp) ** 2
        acc["mm"] += np.abs(rm) ** 2
        acc["pm"] += (rp * np.conjugate(rm)).real
    for k in acc:
        acc[k] /= (nfr * V)

    kmag = np.linalg.norm(kvec, axis=1)
    shells = sorted(set(idx))
    out = {"k": [], "n2": []}
    for k in acc:
        out[k] = []
    for s in shells:
        m = idx == s
        out["k"].append(kmag[m].mean())
        out["n2"].append(s)
        for k in acc:
            out[k].append(acc[k][m].mean())
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="zero-field run directory")
    ap.add_argument("--npt", default=None, help="NPT directory holding vol.dat")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--eps-r", type=float, default=M.EPS_R,
                    help="relative permittivity of the run (LAMMPS `dielectric`)")
    ap.add_argument("--dt", type=float, default=M.DT)
    ap.add_argument("--discard", type=float, default=0.3, help="NPT fraction to discard")
    ap.add_argument("--sk-frames", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1, help="read every Nth dump frame")
    ap.add_argument("--conc", type=float, default=None,
                    help="c_LJ, enables the comparison with the paper's SI figures")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {"temp": a.temp}
    print("=" * 72)

    # --- 1. density from the NPT trace ---------------------------------------
    if a.npt:
        _, v = dumpio.read_ave_time(os.path.join(a.npt, "vol.dat"))
        n0 = int(a.discard * len(v))
        L, ra, rm, pr = (v[n0:, 1], v[n0:, 2], v[n0:, 3], v[n0:, 4])
        # block averages for an honest error bar
        nb = 10
        Lb = np.array([L[i::nb].mean() for i in range(nb)])
        res["L_mean"] = float(L.mean()); res["L_err"] = float(Lb.std(ddof=1) / np.sqrt(nb))
        res["rho_all"] = float(ra.mean()); res["rho_mono"] = float(rm.mean())
        res["press_mean"] = float(pr.mean())
        drift = np.polyfit(np.arange(len(L)), L, 1)[0] * len(L)
        res["L_drift_over_window"] = float(drift)
        print(f"NPT (p=0), discarding the first {a.discard:.0%} of {len(v)} samples")
        print(f"  L        = {L.mean():.4f} +- {res['L_err']:.4f} sigma   "
              f"(drift over window {drift:+.4f})")
        print(f"  rho_all  = {ra.mean():.5f} sigma^-3   rho_mono = {rm.mean():.5f} sigma^-3")
        print(f"  <P>      = {pr.mean():+.5f} eps/sigma^3  (target 0)")

    # --- 2. ion dynamics ------------------------------------------------------
    zf = os.path.join(a.run, "ions.dump")
    if not os.path.exists(zf) and os.path.exists(zf + ".gz"):
        zf += ".gz"
    steps, boxes, cols, data = dumpio.read_dump(zf, stride=a.stride)
    typ = data[0, :, cols.index("type")].astype(int)
    r = np.stack([data[:, :, cols.index(c)] for c in ("xu", "yu", "zu")], axis=2)
    t = (steps - steps[0]) * a.dt
    print(f"\nzero-field trajectory: {len(steps)} frames, {r.shape[1]} ions, "
          f"t_max = {t[-1]:.1f} tau")

    for lab, tt in (("cation", M.T_CAT), ("anion", M.T_ANI)):
        m = typ == tt
        msd = msd_fft(r[:, m, :])
        D = fit_D(t[1:], msd[1:])
        sl = msd_loglog_slope(t[1:], msd[1:])
        res[f"D_{lab}"] = float(D)
        res[f"msd_slope_{lab}"] = float(sl)
        print(f"  D_{lab:7s} = {D:.4e} sigma^2/tau   "
              f"(log-log slope {sl:.3f}; 1.0 = diffusive)")
        np.savetxt(os.path.join(a.run, f"msd_{lab}.dat"),
                   np.c_[t, msd], header="t msd")

    Ds = min(res["D_cation"], res["D_anion"])
    res["D_s"] = float(Ds)

    # --- 3. conductivity / Nernst-Einstein ------------------------------------
    s = np.where(typ == M.T_CAT, 1.0, -1.0)
    Mt = (r * s[None, :, None]).sum(axis=1)                     # charge dipole
    msd_q = msd_fft(Mt[:, None, :])
    msd_sum = sum(msd_fft(r[:, typ == tt, :]) * (typ == tt).sum()
                  for tt in (M.T_CAT, M.T_ANI))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = msd_q / msd_sum
    i0, i1 = int(0.2 * len(t)), int(0.8 * len(t))
    res["sigma_over_NE"] = float(np.nanmean(ratio[i0:i1]))
    print(f"  sigma/sigma_NE = {res['sigma_over_NE']:.3f}   "
          f"(paper's model: near 1; atomistic PEO/LiTFSI: about 0.1)")
    np.savetxt(os.path.join(a.run, "msd_charge.dat"), np.c_[t, msd_q, msd_sum],
               header="t msd_charge msd_sum")

    # --- 4. structure factors and the Debye limit -----------------------------
    L = float(boxes[0][0, 1] - boxes[0][0, 0])
    res["L_prod"] = L
    nfr = min(a.sk_frames, r.shape[0])
    sel = np.linspace(0, r.shape[0] - 1, nfr).astype(int)
    sk = structure_factors(r[sel], s, L, nmax=8)
    n_each = (typ == M.T_CAT).sum() / L ** 3
    lB = M.bjerrum_sigma(a.temp, a.eps_r)
    kD2 = 8.0 * np.pi * lB * n_each
    res["n_each"] = float(n_each); res["lB"] = float(lB)
    res["kappa_D"] = float(np.sqrt(kD2)); res["debye_length"] = float(1.0 / np.sqrt(kD2))
    print(f"\n  n_+ = n_- = {n_each:.5f} sigma^-3, l_B = {lB:.3f} sigma, "
          f"kappa_D^-1 = {1/np.sqrt(kD2):.3f} sigma")
    print("   k       S_ZZ/(2n)     k^2/kD^2      ratio      S_NN/(2n)")
    dev = []
    for i in range(min(8, len(sk["k"]))):
        k = sk["k"][i]
        lhs = sk["ZZ"][i] / (2 * n_each)
        rhs = k ** 2 / kD2
        print(f"  {k:6.3f}  {lhs:11.5f}  {rhs:11.5f}  {lhs/rhs:9.3f}  "
              f"{sk['NN'][i]/(2*n_each):11.5f}")
        if i < 3:
            dev.append(lhs / rhs)
    res["debye_ratio_lowk"] = [float(x) for x in dev]
    np.savetxt(os.path.join(a.run, "sk.dat"),
               np.c_[sk["k"], sk["ZZ"], sk["NN"], sk["pp"], sk["mm"], sk["pm"]],
               header="k S_ZZ S_NN S_++ S_-- S_+-")

    # --- 5. run lengths implied by the measured D_s ---------------------------
    print(f"\nrun lengths implied by D_s = {Ds:.3e} sigma^2/tau (Section 6):")
    for lam in (4.0, 6.0, 8.0):
        tl = lam ** 2 / (4.0 * np.pi ** 2 * Ds)
        print(f"  lambda={lam:4.1f} sigma: tau_lambda = {tl:9.1f} tau"
              + (f"   transient 10x = {10*tl:9.1f} tau   "
                 f"production 100x = {100*tl:10.1f} tau "
                 f"({100*tl/M.DT:.2e} steps)" if lam == 8.0 else ""))
    res["tau_lambda_8"] = float(64.0 / (4 * np.pi ** 2 * Ds))

    # --- 6. acceptance check against the paper -------------------------------
    if a.conc is not None and abs(a.temp - 1.0) < 1e-9:
        e = PR.expected(a.conc)
        res["paper"] = e
        print(f"\nacceptance check vs. Tsamopoulos & Wang SI "
              f"(c_LJ={a.conc}, their c_s={e['c_s']}):")
        print(f"{'quantity':>16} {'measured':>12} {'paper':>12} {'ratio':>8}  status")
        rows = []
        if "L_mean" in res:
            rows.append(("L", res["L_mean"], e["L"], 0.02))
            rows.append(("rho_all", res["rho_all"], e["rho_all"], 0.02))
        rows.append(("D_cation", res["D_cation"], e["D_cation"], 0.25))
        rows.append(("D_anion", res["D_anion"], e["D_anion"], 0.25))
        allok = True
        for name, got, ref, tol in rows:
            r = got / ref
            ok = abs(r - 1.0) <= tol
            allok &= ok
            print(f"{name:>16} {got:12.5g} {ref:12.5g} {r:8.3f}  "
                  f"{'OK' if ok else 'OUT (tol %.0f%%)' % (100*tol)}")
        res["paper_check_passed"] = bool(allok)
        print("  the SI values are digitised from vector figures by eye: "
              "treat them as +-5% targets")
        print(f"  -> paper reproduction check: {'PASS' if allok else 'REVIEW'}")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
