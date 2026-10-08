#!/usr/bin/env python3
"""Nonlinearity of the drive pilots against the parameter-free linear response of the zero-field chain.

Linear response (zero field, exact): a step V_b(z) = A cos(k z) v_b gives dn_a(k, t) = -(A/2) [S - F(k, t)]_ab v_b
(coefficient of exp(+ikz)), v = (1, 1) for V_N and (1, -1) for psi; flux j = i dn/dt / k.
Step runs: odd part = (phase 0 - phase pi)/2 per seed (exact sign flip of V), even part = sum/2 (modes 2m);
ratio = Re(sum meas conj(pred)) / sum |pred|^2 per time window, errors over the 32 seeds.
Early flux (0-2 tau): the density has not moved, the force is the bare external force, so flux/pred is the
force dependence of the friction itself, free of any functional.
Periodic runs: V(t) = A cos(kz) sin(2 pi t/T); steady state after t0 = max(2T, 500); lock-in harmonics h = 1, 3
of the density and flux modes, measured / linear-prediction fundamental, and |h=3| / |h=1|.
Writes analysis/summary.json and prints tables.
"""
import glob
import json
import os

import numpy as np

ROOT = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
D = os.path.join(ROOT, "runs/drive_T1.0_c0.04")
FK = np.load(os.path.join(ROOT, "runs/relax_T1.0_c0.04/zf_seed/Fkt.npz"))
F = FK["F"].mean(1)                                           # [3, M, lag]
L = 24.502285
WIN = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 50), (50, 100), (100, 300)]
LWIN = [(300, 1000), (1000, 3000)]


def Fmat(m, T):
    pp, mm, pm = F[0, m - 1], F[1, m - 1], F[2, m - 1]
    M = np.stack([np.stack([pp, pm], -1), np.stack([pm, mm], -1)], -2)           # [lag, 2, 2]
    if T > M.shape[0]:
        M = np.concatenate([M, np.zeros((T - M.shape[0], 2, 2))])
    return M[:T]


def step_pred(m, A, ch, T):
    v = np.array([1.0, 1.0]) if ch == "N" else np.array([1.0, -1.0])
    Fm = Fmat(m, T)
    n = -(A / 2) * np.einsum("tab,b->ta", Fm[0][None] - Fm, v)                    # [T, 2]
    k = 2 * np.pi * m / L
    j = np.full_like(n, np.nan, dtype=complex)
    j[1:] = 1j * (n[1:] - n[:-1]) / k
    return n.astype(complex), j


def ratio(meas, pred, a, b):
    """meas [R, T] per seed, pred [T]: projection ratio for t in (a, b], mean and SE over seeds."""
    sl = slice(a + 1, b + 1)
    den = np.sum(np.abs(pred[sl]) ** 2)
    r = np.array([np.sum(x[sl] * np.conj(pred[sl])).real / den for x in meas])
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))


def parse(tag):
    m, ch, A = tag.split("_")[:3]
    return int(m[1:]), ch, float(A[1:])


def steps():
    out = {}
    for f in sorted(glob.glob(os.path.join(D, "analysis/step_*.npz"))):
        tag = os.path.basename(f)[5:-4]
        m, ch, A = parse(tag)
        d = np.load(f)
        names = list(d["names"])
        idx0 = [i for i, nm in enumerate(names) if nm.endswith("_ph0")]
        idx1 = [names.index(names[i][:-1] + "1") for i in idx0]
        n0, n1, j0, j1 = d["n"][idx0], d["n"][idx1], d["j"][idx0], d["j"][idx1]
        T = n0.shape[1]
        pn, pj = step_pred(m, A, ch, T)
        r = {"m": m, "channel": ch, "A": A, "pp_kT": 2 * A}
        for s, lab in enumerate(("cation", "anion")):
            odd_n = (n0[:, :, s, 0] - n1[:, :, s, 0]) / 2
            odd_j = (j0[:, :, s] - j1[:, :, s]) / 2
            even2 = (n0[:, :, s, 1] + n1[:, :, s, 1]) / 2
            odd3 = (n0[:, :, s, 2] - n1[:, :, s, 2]) / 2
            r[lab] = {"n_ratio": {f"{a}-{b}": ratio(odd_n, pn[:, s], a, b) for a, b in WIN},
                      "j_ratio": {f"{a}-{b}": ratio(odd_j, pj[:, s], a, b) for a, b in WIN},
                      "harm2_over_1_late": float(np.abs(even2[:, 100:].mean()) / np.abs(odd_n[:, 100:].mean())),
                      "harm3_over_1_late": float(np.abs(odd3[:, 100:].mean()) / np.abs(odd_n[:, 100:].mean()))}
        lf = os.path.join(D, f"analysis/long_{tag}.npz")
        if os.path.exists(lf):
            dl = np.load(lf)
            Tl = dl["n"].shape[1]
            pnl, _ = step_pred(m, A, ch, Tl)
            for s, lab in enumerate(("cation", "anion")):
                r[lab]["n_ratio_long"] = {f"{a}-{b}": ratio(dl["n"][:, :, s, 0], pnl[:, s], a, b) for a, b in LWIN}
        out[tag] = r
    return out


def lockin(x, t, T, h):
    w = 2 * np.pi / T
    return 2 * np.mean(x * np.exp(-1j * h * w * t), axis=-1)


def periodic():
    out = {}
    for f in sorted(glob.glob(os.path.join(D, "analysis/per_*.npz"))):
        tag = os.path.basename(f)[4:-4]
        m, ch, A = parse(tag)
        T = float(tag.split("_T")[-1])
        d = np.load(f)
        n, j = d["n"], d["j"]                                                     # [R, Tt, 2, 3], [R, Tt, 2]
        Tt = n.shape[1]
        t = np.arange(Tt, dtype=float)
        # linear prediction for the driven time series: n(t) = sum_s dR(t - s) u(s + 1/2), R = unit-step response
        R1, _ = step_pred(m, A, ch, Tt + 1)
        dR = np.diff(R1, axis=0)                                                  # [Tt, 2]
        u = np.sin(2 * np.pi * (t + 0.5) / T)
        pn = np.stack([np.convolve(dR[:, s], u)[:Tt] for s in range(2)], -1)
        pn = np.concatenate([np.zeros((1, 2)), pn[:-1]])                           # n(0) = 0
        k = 2 * np.pi * m / L
        pj = np.full_like(pn, np.nan)
        pj[1:] = 1j * (pn[1:] - pn[:-1]) / k
        t0 = int(max(2 * T, 500))
        nper = int((Tt - 1 - t0) // T)
        sl = slice(t0, t0 + int(nper * T))
        ts = t[sl]
        r = {"m": m, "channel": ch, "A": A, "T": T, "periods_used": nper}
        for s, lab in enumerate(("cation", "anion")):
            c1 = lockin(n[:, sl, s, 0], ts, T, 1)
            c3 = lockin(n[:, sl, s, 0], ts, T, 3)
            p1 = lockin(pn[sl, s], ts, T, 1)
            jc1 = lockin(j[:, sl, s], ts - 0.5, T, 1)
            jp1 = lockin(pj[sl, s], ts - 0.5, T, 1)
            rr = (c1 * np.conj(p1)).real / abs(p1) ** 2
            rj = (jc1 * np.conj(jp1)).real / abs(jp1) ** 2
            r[lab] = {"n_fund_ratio": [float(rr.mean()), float(rr.std(ddof=1) / 2)],
                      "n_fund_phase_deg": float(np.degrees(np.angle(c1.mean() / p1))),
                      "j_fund_ratio": [float(rj.mean()), float(rj.std(ddof=1) / 2)],
                      "n_h3_over_h1": float(abs(c3.mean()) / abs(c1.mean()))}
        out[tag] = r
    return out


def main():
    S, P = steps(), periodic()
    json.dump({"step": S, "periodic": P}, open(os.path.join(D, "analysis/summary.json"), "w"), indent=1)
    print("STEP: measured odd response / linear response, per window", [f"{a}-{b}" for a, b in WIN])
    for tag, r in S.items():
        for lab in ("cation", "anion"):
            x = r[lab]
            print(f" {tag:16s} {lab[:3]} n: " + " ".join(f"{v[0]:5.2f}" for v in x["n_ratio"].values()) +
                  " | j: " + " ".join(f"{v[0]:5.2f}" for v in x["j_ratio"].values()) +
                  (" | long n: " + " ".join(f"{v[0]:5.2f}" for v in x.get("n_ratio_long", {}).values())) +
                  f" | h2/h1 {x['harm2_over_1_late']:.3f} h3/h1 {x['harm3_over_1_late']:.3f}")
    print("PERIODIC: fundamental measured / linear (n, j), phase, h3/h1")
    for tag, r in P.items():
        print(f" {tag:22s} " + " | ".join(f"{lab[:3]} n {r[lab]['n_fund_ratio'][0]:.2f}±{r[lab]['n_fund_ratio'][1]:.2f}"
                                          f" ph {r[lab]['n_fund_phase_deg']:+.0f} j {r[lab]['j_fund_ratio'][0]:.2f}"
                                          f" h3 {r[lab]['n_h3_over_h1']:.3f}" for lab in ("cation", "anion")))


if __name__ == "__main__":
    main()
