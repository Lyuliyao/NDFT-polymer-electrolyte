"""Streaming reader for LAMMPS text dumps, with a .npz cache.

The dumps written by this campaign have a fixed atom count per frame and are
sorted by id, so every frame is a fixed-size block of numbers.
"""
import gzip
import os
import sys

import numpy as np


def _open(fn):
    return gzip.open(fn, "rt") if fn.endswith(".gz") else open(fn)


def _frames(fn):
    """Yield (timestep, box(3,2), columns, ndarray(natoms, ncol))."""
    with _open(fn) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()                      # ITEM: NUMBER OF ATOMS
            nat = int(fh.readline())
            fh.readline()                      # ITEM: BOX BOUNDS
            box = np.array([[float(x) for x in fh.readline().split()[:2]] for _ in range(3)])
            cols = fh.readline().split()[2:]   # ITEM: ATOMS id type ...
            buf = [fh.readline() for _ in range(nat)]
            arr = np.fromstring("".join(buf), sep=" ").reshape(nat, len(cols))
            yield step, box, cols, arr


def read_dump(fn, cache=True, stride=1):
    """Read a dump into (steps, boxes, cols, data[nframe, natom, ncol]).

    Atoms are ordered by id (the dumps are written with `dump_modify sort id`).
    """
    cf = fn + (f".s{stride}.npz" if stride != 1 else ".npz")
    if cache and os.path.exists(cf) and os.path.getmtime(cf) > os.path.getmtime(fn):
        z = np.load(cf, allow_pickle=True)
        return z["steps"], z["boxes"], list(z["cols"]), z["data"]

    steps, boxes, data, cols = [], [], [], None
    for k, (st, bx, cl, ar) in enumerate(_frames(fn)):
        if k % stride:
            continue
        cols = cl
        steps.append(st); boxes.append(bx); data.append(ar)
    steps = np.array(steps)
    boxes = np.array(boxes)
    data = np.array(data)
    if cache:
        try:
            np.savez_compressed(cf, steps=steps, boxes=boxes, cols=np.array(cols), data=data)
        except OSError as e:          # out of quota/space: the cache is optional
            print(f"warning: could not write {cf}: {e}", file=sys.stderr)
            try:
                os.remove(cf)
            except OSError:
                pass
    return steps, boxes, cols, data


def col(cols, name):
    return cols.index(name)


def read_ave_time(fn):
    """Read a `fix ave/time` file into (header, ndarray)."""
    rows, head = [], []
    for ln in open(fn):
        if ln.startswith("#"):
            head.append(ln.strip())
            continue
        rows.append([float(x) for x in ln.split()])
    return head, np.array(rows)


def read_ave_chunk(fn):
    """Read a `fix ave/chunk` file.

    Returns (steps[nblock], coord[nbin], values[nblock, nbin, nval], names).
    """
    names = None
    steps, blocks = [], []
    cur = None
    with open(fn) as fh:
        for ln in fh:
            if ln.startswith("#"):
                if "Chunk" in ln:
                    names = ln.strip("#").split()[1:]
                continue
            p = ln.split()
            if len(p) == 3 and "." not in p[0]:         # block header: step nchunks ntotal
                if cur is not None:
                    blocks.append(cur)
                steps.append(int(p[0]))
                cur = []
            else:
                cur.append([float(x) for x in p])
    if cur is not None:
        blocks.append(cur)
    arr = np.array(blocks)                              # (nblock, nbin, ncol)
    coord = arr[0, :, 1]
    return np.array(steps), coord, arr[:, :, 2:], (names[2:] if names else None)
