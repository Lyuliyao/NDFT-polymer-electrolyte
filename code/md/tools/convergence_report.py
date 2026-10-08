#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio
import model as M
import paper_reference as PR
from analyze_zerofield import msd_fft


def block_se(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def fit_slope(t, y, lo, hi):
    m = (t >= lo) & (t <= hi)
    return np.polyfit(t[m], y[m], 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--conc", type=float, default=0.04)
    ap.add_argument("--nblock", type=int, default=4)
    ap.add_argument("--discard", type=float, default=0.3, help="NPT fraction to discard")
    a = ap.parse_args()
    ref = PR.expected(a.conc)
    res = {}


    d = np.loadtxt(os.path.join(a.state, "equil", "vol.dat"), comments="#")
    tn = (d[:, 0] - d[0, 0]) * M.DT
    n0 = int(a.discard * len(d))
    V = d[n0:, 1] ** 3
    h = len(V) // 2
    Vb = np.array([b.mean() for b in np.array_split(V, 10)])
    drift = np.polyfit(tn[n0:], V, 1)[0] * (tn[-1] - tn[n0])
    res.update(V=V.mean(), V_se=Vb.std(ddof=1) / np.sqrt(len(Vb)),
               V_half1=V[:h].mean(), V_half2=V[h:].mean(), V_drift=drift,
               P=d[n0:, 4].mean(), npt_tau=tn[-1])
    print("=" * 74)
    print(f"NPT  {tn[-1]:.0f} tau, averaging the last {1-a.discard:.0%}")
    print(f"  V = {res['V']:.1f} +- {res['V_se']:.1f}   halves {res['V_half1']:.1f} / "
          f"{res['V_half2']:.1f}   linear drift over window {drift:+.1f}   <P> = {res['P']:+.4f}")
    print(f"  paper {ref['V']:.0f}  -> ratio {res['V']/ref['V']:.4f}")


    zf = os.path.join(a.state, "zerofield", "ions.dump")
    steps, boxes, cols, data = dumpio.read_dump(zf)
    typ = data[0, :, cols.index("type")].astype(int)
    r = np.stack([data[:, :, cols.index(c)] for c in ("xu", "yu", "zu")], axis=2)
    del data
    t = (steps - steps[0]) * M.DT
    com = r.mean(axis=1)
    vcom = np.linalg.norm(np.polyfit(t, com, 1)[0])
    r -= com[:, None, :]
    Vbox = float(np.mean([np.prod(b[:, 1] - b[:, 0]) for b in boxes[:1]]))
    cat, ani = typ == M.T_CAT, typ == M.T_ANI
    npair = int(cat.sum())
    res.update(zf_tau=t[-1], v_com=vcom, V_box=Vbox)
    print(f"\nzero-field  {t[-1]:.0f} tau, {len(t)} frames, {npair} pairs, box V = {Vbox:.1f}")
    print(f"  system COM drift {vcom:.2e} sigma/tau (removed; before the 2026-09-22 fix ~1e-2)")


    print("  local log-log MSD slope over lag windows (1 = diffusive):")
    wins = [w for w in ((10, 100), (100, 1000), (1000, 3000), (3000, 6000))
            if w[1] <= 0.5 * t[-1]]
    msd_full = {}
    for lab, m in (("cation", cat), ("anion", ani)):
        msd_full[lab] = msd_fft(r[:, m, :])
        sl = []
        for lo, hi in wins:
            k = (t >= lo) & (t <= hi)
            sl.append(np.polyfit(np.log(t[k]), np.log(msd_full[lab][k]), 1)[0])
            res[f"slope_{lab}_{lo}_{hi}"] = sl[-1]
        print(f"    {lab:7s} " + "  ".join(f"[{lo:>4d},{hi:>4d}] {s:.3f}"
                                           for (lo, hi), s in zip(wins, sl)))


    nb = a.nblock
    edges = np.linspace(0, len(t), nb + 1).astype(int)
    Dc, Da, sig, ratio = [], [], [], []
    for b in range(nb):
        sl_ = slice(edges[b], edges[b + 1])
        tb = t[sl_] - t[sl_][0]
        lo, hi = 0.2 * tb[-1], 0.6 * tb[-1]
        rb = r[sl_]
        mc = msd_fft(rb[:, cat, :]); ma = msd_fft(rb[:, ani, :])
        s = np.where(cat, 1.0, -1.0)
        mq = msd_fft((rb * s[None, :, None]).sum(axis=1)[:, None, :])
        Dc.append(fit_slope(tb, mc, lo, hi) / 6.0)
        Da.append(fit_slope(tb, ma, lo, hi) / 6.0)
        sg = fit_slope(tb, mq, lo, hi) / (6.0 * Vbox)
        sig.append(sg)
        ratio.append(sg / (npair * (Dc[-1] + Da[-1]) / Vbox))
    blocks = dict(D_cation=Dc, D_anion=Da, sigma=sig, sigma_over_NE=ratio)
    res["blocks"] = blocks
    print(f"\n  {nb} blocks of {t[-1]/nb:.0f} tau (D fitted over lags 0.2-0.6 of a block):")
    print(f"  {'quantity':>14s} " + " ".join(f"{'blk'+str(i):>10s}" for i in range(nb))
          + f" {'mean':>10s} {'+- se':>9s} {'paper':>10s} {'ratio':>7s}")
    refs = dict(D_cation=ref["D_cation"], D_anion=ref["D_anion"], sigma=PR.SIGMA_T1,
                sigma_over_NE=None)
    for k, v in blocks.items():
        mu, se = block_se(v)
        res[k], res[k + "_se"] = mu, se
        rf = refs[k]
        print(f"  {k:>14s} " + " ".join(f"{x:10.3e}" for x in v) + f" {mu:10.3e} {se:9.1e}"
              + (f" {rf:10.3e} {mu/rf:7.3f}" if rf else ""))
    mu_r = res["D_anion"] / res["D_cation"]
    print(f"  {'D-/D+':>14s} {mu_r:.3f}   (paper {ref['D_anion']/ref['D_cation']:.3f})")
    res["Dratio"] = mu_r
    print("  paper conductivity is an order-of-magnitude reading of Fig. S7")
    print("=" * 74)

    out = os.path.join(a.state, "convergence.json")
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
