import gzip
import os
import sys

import numpy as np


def _open(fn):
    return gzip.open(fn, "rt") if fn.endswith(".gz") else open(fn)


def _frames(fn):

    with _open(fn) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()
            nat = int(fh.readline())
            fh.readline()
            box = np.array([[float(x) for x in fh.readline().split()[:2]] for _ in range(3)])
            cols = fh.readline().split()[2:]
            buf = [fh.readline() for _ in range(nat)]
            arr = np.fromstring("".join(buf), sep=" ").reshape(nat, len(cols))
            yield step, box, cols, arr


def read_dump(fn, cache=True, stride=1):

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
        except OSError as e:
            print(f"warning: could not write {cf}: {e}", file=sys.stderr)
            try:
                os.remove(cf)
            except OSError:
                pass
    return steps, boxes, cols, data


def col(cols, name):
    return cols.index(name)


def read_ave_time(fn):

    rows, head = [], []
    for ln in open(fn):
        if ln.startswith("#"):
            head.append(ln.strip())
            continue
        rows.append([float(x) for x in ln.split()])
    return head, np.array(rows)


def read_ave_chunk(fn):

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
            if len(p) == 3 and "." not in p[0]:
                if cur is not None:
                    blocks.append(cur)
                steps.append(int(p[0]))
                cur = []
            else:
                cur.append([float(x) for x in p])
    if cur is not None:
        blocks.append(cur)
    arr = np.array(blocks)
    coord = arr[0, :, 1]
    return np.array(steps), coord, arr[:, :, 2:], (names[2:] if names else None)
