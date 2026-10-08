#!/usr/bin/env python3
"""Reduce the drive pilot runs to mode time series (runs/drive_T1.0_c0.04).

    drive_extract.py [--nproc 4]

For every run: particle-exact density modes n_a(h m, t) = (1/V) sum_i exp(-i h k_m z_i), h = 1, 2, 3, on the
dump frames (t = 0, 1, ... tau), and the per-step binned flux mode j_a(m) (window-mean n<v_z>, bin-centre
modes divided by sinc), per species a.  Output per potential directory: analysis/<set>_<tag>.npz with
n[run, t, species, h], j[run, t, species], names.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import dumpio  # noqa: E402

D = os.path.join(ROOT, "runs/drive_T1.0_c0.04")
L = 24.502285


def one(args):
    rundir, m = args
    k = 2 * np.pi * m / L
    V = L ** 3
    n, j = [], []
    for sp in ("cat", "ani"):
        st, bx, cols, data = dumpio.read_dump(os.path.join(rundir, f"{sp}.dump"), cache=False)
        z = data[:, :, cols.index("zu")]
        n.append(np.stack([np.exp(-1j * h * k * z).sum(1) / V for h in (1, 2, 3)], -1))      # [T, 3]
        parts = []
        for f in sorted(glob.glob(os.path.join(rundir, f"ave_{sp}.*.dat")), key=lambda x: int(x.split(".")[-2])):
            s, coord, v, _ = dumpio.read_ave_chunk(f)
            parts.append((s, v))
        s, v = parts[0]
        for s2, v2 in parts[1:]:
            keep = s < s2[0]
            s, v = np.concatenate([s[keep], s2]), np.concatenate([v[keep], v2])
        jb = v[:, :, 1] * v[:, :, 2]                                                          # [W, Nb]
        Nb = jb.shape[1]
        jm = np.fft.rfft(jb, axis=1)[:, m] / Nb * np.exp(-1j * k * (L / Nb) / 2) / np.sinc(k * (L / Nb) / 2 / np.pi)
        j.append(np.concatenate([[np.nan], jm]))                                              # window k at index k
    T = min(len(n[0]), len(j[0]))
    return np.stack([x[:T] for x in n], 1), np.stack([x[:T] for x in j], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=4)
    a = ap.parse_args()
    out = os.path.join(D, "analysis")
    os.makedirs(out, exist_ok=True)
    groups = {}
    for lst, kind in (("step_runs.txt", "step"), ("long_runs.txt", "long"), ("per_runs.txt", "per")):
        for line in open(os.path.join(D, lst)):
            r = line.split()[0]
            r = r if r.startswith("/") else os.path.join(ROOT, r)
            tag = os.path.basename(os.path.dirname(r))
            groups.setdefault((kind, tag), []).append(r)
    with Pool(a.nproc) as pool:
        for (kind, tag), runs in sorted(groups.items()):
            f = os.path.join(out, f"{kind}_{tag}.npz")
            if os.path.exists(f):
                continue
            m = int(tag.split("_")[0][1:])
            res = pool.map(one, [(r, m) for r in runs])
            T = min(x[0].shape[0] for x in res)
            np.savez(f, n=np.stack([x[0][:T] for x in res]), j=np.stack([x[1][:T] for x in res]),
                     names=np.array([os.path.relpath(r, D) for r in runs]), m=m, L=L)
            print(f, len(runs), T, flush=True)


if __name__ == "__main__":
    main()
