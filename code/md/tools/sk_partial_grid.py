#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio
import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zf", required=True)
    ap.add_argument("--lz", type=float, required=True)
    ap.add_argument("--mmax", type=int, default=27)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--nblock", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps, boxes, cols, data = dumpio.read_dump(os.path.join(a.zf, "ions.dump"))
    sel = np.linspace(0, len(steps) - 1, min(a.frames, len(steps))).astype(int)
    typ = data[0, :, cols.index("type")].astype(int)
    pos = data[sel][:, :, [cols.index(c) for c in ("xu", "yu", "zu")]]
    L = a.lz; V = L ** 3
    cat, ani = typ == M.T_CAT, typ == M.T_ANI
    m = np.arange(1, a.mmax + 1); k = 2 * np.pi * m / L
    Spp = np.zeros((len(sel), len(m))); Smm = np.zeros_like(Spp); Spm = np.zeros_like(Spp)
    for ax in range(3):
        ph = np.exp(-1j * pos[:, :, ax][:, :, None] * k[None, None, :])
        rp, rm = ph[:, cat].sum(1), ph[:, ani].sum(1)
        Spp += np.abs(rp) ** 2 / V / 3; Smm += np.abs(rm) ** 2 / V / 3; Spm += (rp * np.conj(rm)).real / V / 3
    out = dict(m=m, k=k, L=L, n_frames=len(sel), n_each=cat.sum() / V)
    for lab, S in (("S_pp", Spp), ("S_mm", Smm), ("S_pm", Spm)):
        blocks = np.array_split(S, a.nblock)
        bm = np.array([b.mean(0) for b in blocks])
        out[lab] = S.mean(0); out[lab + "_err"] = bm.std(0, ddof=1) / np.sqrt(a.nblock)
    np.savez(a.out, **out)
    print(f"{a.zf}: {len(sel)} frames, n_each {out['n_each']:.5f}; S_pp/n at k1 {out['S_pp'][0]/out['n_each']:.3f}"
          f" +- {out['S_pp_err'][0]/out['n_each']:.3f}, at k_max {out['S_pp'][-1]/out['n_each']:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
