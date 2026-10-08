#!/usr/bin/env python3

import argparse
import sys

import numpy as np


def read_data(fn):
    lines = open(fn).read().splitlines()
    box = {}
    for ln in lines[:40]:
        t = ln.split()
        if len(t) >= 4 and t[2] in ("xlo", "ylo", "zlo"):
            box[t[2][0]] = (float(t[0]), float(t[1]))
    i0 = next(i for i, l in enumerate(lines) if l.startswith("Atoms"))
    at = []
    for l in lines[i0 + 2:]:
        t = l.split()
        if not t:
            break
        at.append([int(t[2]), float(t[4]), float(t[5]), float(t[6])])
    at = np.array(at)
    return at[:, 0].astype(int), at[:, 1:4], np.array([box[k][1] - box[k][0] for k in "xyz"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--dr", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    edges = np.arange(0, 5.0 + 1e-9, a.dr)
    rc = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    acc = {k: np.zeros(len(rc)) for k in ("cm", "ca", "am")}
    coord = []
    for fn in a.data:
        typ, X, Lv = read_data(fn)
        V = Lv.prod()
        grp = {1: X[typ == 1], 2: X[typ == 2], 3: X[typ == 3]}
        for key, (ia, ib) in {"cm": (2, 1), "ca": (2, 3), "am": (3, 1)}.items():
            A, B = grp[ia], grp[ib]
            h = np.zeros(len(rc))
            for p in A:
                d = B - p
                d -= Lv * np.round(d / Lv)
                r = np.sqrt((d * d).sum(1))
                h += np.histogram(r, bins=edges)[0]
                if key == "cm":
                    coord.append((r < 1.1).sum())
            acc[key] += h / (len(A) * shell * len(B) / V)
    for k in acc:
        acc[k] /= len(a.data)

    print(f"g(r) from {len(a.data)} snapshot(s), bin {a.dr}")
    for key, lab in (("cm", "cation-polymer"), ("ca", "cation-anion"), ("am", "anion-polymer")):
        g = acc[key]; k = np.argmax(g)
        print(f"  {lab:15s} first peak g = {g[k]:6.2f} at r = {rc[k]:.2f}")
    print(f"  cation first-shell coordination (r < 1.1): {np.mean(coord):.2f} monomers")
    print("  paper Fig. S3 (T unstated; c_s = 0.02 / 0.16):")
    print("    cation-polymer ~11.4 / ~12.7 at 0.7;  cation-anion ~2.4 / ~2.1 at ~1.8-1.9")
    if a.out:
        np.savetxt(a.out, np.c_[rc, acc["cm"], acc["ca"], acc["am"]],
                   header="r g_cat_mono g_cat_ani g_ani_mono")


if __name__ == "__main__":
    main()
