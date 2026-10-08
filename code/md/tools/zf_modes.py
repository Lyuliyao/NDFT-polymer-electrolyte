#!/usr/bin/env python3
"""Fourier modes of the ion densities from a zero-field dump (ions only: types 2 = cation, 3 = anion).
For every k = 2 pi n / L with 0 < |n|^2 <= n2max (half space), rho_a(k, t) = sum_{j in a} exp(-i k.r_j) per frame.
Streams the text dump (no cache).  Writes modes.npz: steps, L, N (2,), nvec (nk, 3), n2 (nk,), rho (nt, nk, 2) complex64.
    zf_modes.py --run <replica dir> [--dump ions.dump] [--n2max 9]"""
import argparse, os
import numpy as np


def frames(path):
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline()); fh.readline(); n = int(fh.readline())
            fh.readline(); box = [list(map(float, fh.readline().split()[:2])) for _ in range(3)]
            cols = fh.readline().split()[2:]
            a = np.array([fh.readline().split() for _ in range(n)], float)
            yield step, np.array(box), cols, a


def kvectors(n2max):
    r = int(np.floor(np.sqrt(n2max))); out = []
    for nx in range(0, r + 1):
        for ny in range(-r, r + 1):
            for nz in range(-r, r + 1):
                if nx == 0 and (ny < 0 or (ny == 0 and nz <= 0)):
                    continue
                n2 = nx * nx + ny * ny + nz * nz
                if 0 < n2 <= n2max:
                    out.append((n2, nx, ny, nz))
    out.sort()
    return np.array([o[1:] for o in out], float), np.array([o[0] for o in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--dump", default="ions.dump"); ap.add_argument("--n2max", type=int, default=9)
    a = ap.parse_args()
    nvec, n2 = kvectors(a.n2max)
    steps, rho, L, N = [], [], None, None
    for step, box, cols, d in frames(os.path.join(a.run, a.dump)):
        if L is None:
            L = float(box[0, 1] - box[0, 0])
            kvec = 2 * np.pi * nvec / L
            typ = d[:, cols.index("type")].astype(int); sel = [typ == 2, typ == 3]; N = np.array([s.sum() for s in sel])
        pos = d[:, [cols.index(c) for c in ("xu", "yu", "zu")]]
        ph = np.exp(-1j * (pos @ kvec.T))                                   # (natoms, nk)
        rho.append(np.stack([ph[s].sum(0) for s in sel], -1).astype(np.complex64))
        steps.append(step)
    rho = np.array(rho); steps = np.array(steps)
    # a resumed run may repeat frames (it appends after the checkpoint): keep the last occurrence of each step
    last = {st: i for i, st in enumerate(steps)}; keep = np.array(sorted(last.values()))
    if len(keep) < len(steps):
        print(f"dropping {len(steps) - len(keep)} repeated frames")
    steps, rho = steps[keep], rho[keep]; o = np.argsort(steps); steps, rho = steps[o], rho[o]
    np.savez_compressed(os.path.join(a.run, "modes.npz"), steps=steps, L=L, N=N, nvec=nvec, n2=n2, rho=rho)
    V = L ** 3; s1 = n2 == 1
    snn = (np.abs(rho[:, s1, 0] + rho[:, s1, 1]) ** 2).mean(1) / V / (2 * N[0] / V)
    print(f"{a.run}: {len(steps)} frames, L {L:.4f}, N {N.tolist()}, S_NN(k1)/(2n) = {snn.mean():.4f}  (Gamma {1 / snn.mean():.3f})")


if __name__ == "__main__":
    main()
