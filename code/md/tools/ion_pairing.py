#!/usr/bin/env python3
"""Ion association from a zero-field ions.dump.

A cation and an anion are paired when their distance is below r_c, the first
minimum of g_+-(r) after the contact peak (or --rc).  Averaged over all
frames after --skip, with 10-block standard errors:
  free_+ / free_-   fraction of cations / anions with no counter-ion within r_c
  free              fraction of all ions that are free
  cluster sizes     connected components of the +- contact graph
The +- contact is the only association that matters here (like charges do not
approach at l_B ~ 8 sigma).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio
import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--skip", type=float, default=0.1, help="fraction of frames dropped at the start")
    ap.add_argument("--rc", type=float, default=None, help="pair cutoff; default: first min of g_+-")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    steps, boxes, cols, data = dumpio.read_dump(a.dump, stride=a.stride)
    n0 = int(a.skip * len(steps))
    steps, boxes, data = steps[n0:], boxes[n0:], data[n0:]
    typ = data[0, :, cols.index("type")].astype(int)
    ic, ia = np.where(typ == M.T_CAT)[0], np.where(typ == M.T_ANI)[0]
    xyz = data[:, :, [cols.index(c) for c in ("xu", "yu", "zu")]]
    L = boxes[:, :, 1] - boxes[:, :, 0]

    # g_+-(r)
    rmax, nb = 4.0, 200
    edges = np.linspace(0, rmax, nb + 1)
    h = np.zeros(nb)
    for f in range(len(steps)):
        Lf = L[f]
        d = xyz[f, ic][:, None, :] - xyz[f, ia][None, :, :]
        d -= Lf * np.round(d / Lf)
        r = np.sqrt((d ** 2).sum(-1)).ravel()
        h += np.histogram(r, edges)[0]
    rr = 0.5 * (edges[1:] + edges[:-1])
    V = np.prod(L, axis=1).mean()
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = h / (len(steps) * len(ic) * len(ia) / V * shell)
    ipk = int(np.argmax(g))
    if a.rc is None:
        # minimum of the smoothed g within 0.8 sigma beyond its highest peak
        gs = np.convolve(g, np.ones(7) / 7, mode="same")
        ipk = int(np.argmax(gs))
        w = int(0.8 / (rmax / nb))
        j = ipk + int(np.argmin(gs[ipk:ipk + w]))
        rc = float(rr[j])
    else:
        rc = a.rc

    fp, fm, sizes = [], [], []
    for f in range(len(steps)):
        Lf = L[f]
        pos = np.mod(xyz[f], Lf)
        tc, ta = cKDTree(pos[ic], boxsize=Lf), cKDTree(pos[ia], boxsize=Lf)
        m = tc.sparse_distance_matrix(ta, rc, output_type="coo_matrix")
        nc, na = len(ic), len(ia)
        fp.append(np.mean(np.bincount(m.row, minlength=nc) == 0))
        fm.append(np.mean(np.bincount(m.col, minlength=na) == 0))
        adj = coo_matrix((np.ones(m.nnz), (m.row, m.col + nc)), shape=(nc + na, nc + na))
        _, lab = connected_components(adj, directed=False)
        sizes.append(np.bincount(lab))
    fp, fm = np.array(fp), np.array(fm)
    s = np.concatenate(sizes)
    # fraction of ions in clusters of size 1, 2, 3, >=4
    w = {k: float(s[s == k].sum() / s.sum()) for k in (1, 2, 3)}
    w[">=4"] = float(s[s >= 4].sum() / s.sum())

    def blk(x, nb=10):
        b = np.array([x[i * len(x) // nb:(i + 1) * len(x) // nb].mean() for i in range(nb)])
        return float(x.mean()), float(b.std(ddof=1) / np.sqrt(nb))

    res = dict(dump=a.dump, nframes=int(len(steps)), rc=rc, g_peak_r=float(rr[ipk]),
               g_peak=float(g[ipk]), g_at_rc=float(np.interp(rc, rr, g)),
               free_cation=blk(fp), free_anion=blk(fm), free_all=blk(0.5 * (fp + fm)),
               ion_fraction_in_cluster_size=w)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(dict(res, g_r=rr.tolist(), g_pm=g.tolist()), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
