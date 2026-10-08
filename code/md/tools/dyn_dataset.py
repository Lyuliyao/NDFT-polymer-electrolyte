#!/usr/bin/env python3
"""Dynamical training data from existing equilibrium trajectories (03h 6.11).

    dyn_dataset.py fkt --zf <zero_field dir>      -> <zf>/dyn_F.npz
    dyn_dataset.py cmm --run <field_pXX dir>      -> <run>/dyn_C.npz

fkt: partial intermediate scattering functions of a zero-field run,
    F_ab(k, t) = < rho_a(k, t0 + t) rho_b(k, t0)^* > / V,   rho_a(k) = sum_{i in a} exp(-i k.r_i),
    |k| = 2 pi m / L, m = 1..MMAX, averaged over every lattice vector with n.n = m^2 (isotropy),
    lags 0..LMAX frames, F_+- symmetrized, real part.  S_ab(k) = F_ab(k, 0).
cmm: fluctuation correlations about the equilibrium profile of a planar field run, z modes only
    (k_perp = 0), m = 1..MMAX (m = 0 is fixed by the particle number of each species):
        drho_a(m, t) = rho_a(k_m, t) - <rho_a(k_m)>_run,           k_m = 2 pi m / L along z
        Cp[a, b, m, m', t] = < drho_a(m, t0 + t) drho_b(m', t0)^* > / V
        Cm[a, b, m, m', t] = < drho_a(m, t0 + t) drho_b(m', t0)   > / V   (couples m to -m')
    so C between any signed modes is available (rho(-m) = rho(m)^*).  Translation invariance
    is broken by the profile, so off-diagonal m != m' entries are non-zero.  Stored on a lag
    grid (every frame to 50, every 5 to 300, then log-spaced to LMAX), per block.
Normalization: C = V <dn dn^*> with n_m = rho_m / V, the convention of Sk_grid.npz and Fkt.npz;
at zero field Cp[a, b, m, m, t] = F_ab(k_m, t) for k along z.
Errors: NBLOCK consecutive blocks (the means are subtracted over the whole run).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio  # noqa: E402

T_CAT, T_ANI = 2, 3
CHUNK = 16


def lag_grid(lmax):
    g = list(range(0, 51)) + list(range(55, 301, 5))
    g += list(np.unique(np.round(np.geomspace(320, lmax, 30)).astype(int)))
    return np.array(sorted(set(x for x in g if x <= lmax)))


def load(dumpfile):
    cache = os.path.exists(dumpfile + ".npz")
    steps, boxes, cols, data = dumpio.read_dump(dumpfile, cache=cache)
    d = np.diff(steps)
    assert len(set(d.tolist())) == 1, f"{dumpfile}: uneven frame spacing {sorted(set(d.tolist()))[:5]}"
    L = float(boxes[0][2, 1] - boxes[0][2, 0])
    return steps, L, cols, data


def xcorr(a, b, lags, conj_b=True):
    """(1/n_origins) sum_t a(t+l) b(t)^* (or b(t) if not conj_b); a, b [T, P] -> [len(lags), P]."""
    T = a.shape[0]
    nfft = 1 << (2 * T - 1).bit_length()
    fa = np.fft.fft(a, n=nfft, axis=0)
    fb = np.fft.fft(b if conj_b else np.conj(b), n=nfft, axis=0)
    c = np.fft.ifft(fa * np.conj(fb), axis=0)[lags]
    return c / (T - lags)[:, None]


# ---------------------------------------------------------------------------- fkt
def lattice_vectors(mmax):
    vecs, ms = [], []
    r = range(-mmax, mmax + 1)
    for nx in r:
        for ny in r:
            for nz in r:
                s = nx * nx + ny * ny + nz * nz
                m = int(round(np.sqrt(s)))
                if s == 0 or m * m != s or m > mmax:
                    continue
                if next(c for c in (nx, ny, nz) if c != 0) > 0:
                    vecs.append((nx, ny, nz))
                    ms.append(m)
    o = np.argsort(ms, kind="stable")
    return np.array(vecs)[o], np.array(ms)[o]


def rho_vectors(pos, L, vecs):
    k1 = 2 * np.pi / L
    cmax = int(np.abs(vecs).max())
    c = np.arange(cmax + 1)
    out = np.empty((pos.shape[0], len(vecs)), np.complex64)
    for t0 in range(0, pos.shape[0], CHUNK):
        p = pos[t0:t0 + CHUNK]
        tabs = []
        for dim in range(3):
            e = np.exp(-1j * k1 * p[:, :, dim, None] * c[None, None, :])
            tabs.append(np.concatenate([np.conj(e[:, :, :0:-1]), e], axis=2))
        a = tabs[0][:, :, cmax + vecs[:, 0]]
        a *= tabs[1][:, :, cmax + vecs[:, 1]]
        a *= tabs[2][:, :, cmax + vecs[:, 2]]
        out[t0:t0 + CHUNK] = a.sum(1)
    return out


def cmd_fkt(a):
    t0 = time.time()
    steps, L, cols, data = load(os.path.join(a.zf, "ions.dump"))
    V = L ** 3
    typ = data[0, :, cols.index("type")].astype(int)
    pc = [cols.index(c) for c in ("xu", "yu", "zu")]
    vecs, ms = lattice_vectors(a.mmax)
    rho = {sp: rho_vectors(data[:, typ == ty][:, :, pc], L, vecs) for sp, ty in (("cat", T_CAT), ("ani", T_ANI))}
    nion = {sp: int(np.sum(typ == ty)) for sp, ty in (("cat", T_CAT), ("ani", T_ANI))}
    del data
    T = len(steps)
    lmax = min(a.lmax, T // a.nblock - 1)
    edges = np.linspace(0, T, a.nblock + 1).astype(int)
    lags = np.arange(lmax + 1)
    F = np.zeros((3, a.nblock, a.mmax, lmax + 1))
    for b in range(a.nblock):
        sl = slice(edges[b], edges[b + 1])
        A, B = rho["cat"][sl].astype(np.complex128), rho["ani"][sl].astype(np.complex128)
        cs = (xcorr(A, A, lags).real, xcorr(B, B, lags).real, 0.5 * (xcorr(A, B, lags) + xcorr(B, A, lags)).real)
        for p, c in enumerate(cs):
            for m in range(1, a.mmax + 1):
                F[p, b, m - 1] = c[:, ms == m].mean(1) / V
    out = os.path.join(a.zf, "dyn_F.npz")
    np.savez(out, F=F, m=np.arange(1, a.mmax + 1), k=2 * np.pi * np.arange(1, a.mmax + 1) / L,
             lag_frames=lags, frame_steps=int(steps[1] - steps[0]), L=L, n_cat=nion["cat"] / V,
             n_ani=nion["ani"] / V, nvec=np.bincount(ms, minlength=a.mmax + 1)[1:], pairs=np.array(["++", "--", "+-"]),
             frames=T, nblock=a.nblock)
    print(f"{out}: {T} frames, L={L:.4f}, N={nion}, lmax={lmax}, {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------- cmm
def rho_z(z, L, mmax):
    """z [T, N] -> rho [T, mmax] = sum_i exp(-i k_m z_i), m = 1..mmax."""
    k1 = 2 * np.pi / L
    out = np.empty((z.shape[0], mmax), np.complex128)
    m = np.arange(1, mmax + 1)
    for t0 in range(0, z.shape[0], 256):
        e = np.exp(-1j * k1 * z[t0:t0 + 256, :, None] * m[None, None, :])
        out[t0:t0 + 256] = e.sum(1)
    return out


def cmd_cmm(a):
    t0 = time.time()
    rho, info = [], {}
    for sp in ("cat", "ani"):
        steps, L, cols, data = load(os.path.join(a.run, f"{sp}.dump"))
        zc = "zu" if "zu" in cols else "z"
        rho.append(rho_z(data[:, :, cols.index(zc)], L, a.mmax))
        info[sp] = (len(steps), data.shape[1], int(steps[1] - steps[0]), int(steps[0]))
        del data
    assert info["cat"][0] == info["ani"][0], info
    V = L ** 3
    X = np.concatenate(rho, axis=1)                                     # [T, 2M]: cations m=1..M, anions m=1..M
    mean = X.mean(0)
    dX = X - mean
    T = X.shape[0]
    lmax = min(a.lmax, T // a.nblock - 1)
    lags = lag_grid(lmax)
    edges = np.linspace(0, T, a.nblock + 1).astype(int)
    M2 = X.shape[1]
    Cp = np.zeros((a.nblock, len(lags), M2, M2), np.complex64)
    Cm = np.zeros((a.nblock, len(lags), M2, M2), np.complex64)
    for b in range(a.nblock):
        Y = dX[edges[b]:edges[b + 1]]
        nfft = 1 << (2 * len(Y) - 1).bit_length()
        fy = np.fft.fft(Y, n=nfft, axis=0)                              # [nfft, 2M]
        fyc = np.fft.fft(np.conj(Y), n=nfft, axis=0)
        n_or = (len(Y) - lags)[:, None, None]
        for j in range(M2):                                             # column = the earlier-time mode
            cp = np.fft.ifft(fy * np.conj(fy[:, j:j + 1]), axis=0)[lags]
            cm = np.fft.ifft(fy * np.conj(fyc[:, j:j + 1]), axis=0)[lags]
            Cp[b, :, :, j] = (cp / n_or[:, :, 0]) / V
            Cm[b, :, :, j] = (cm / n_or[:, :, 0]) / V
    M = a.mmax
    rs = lambda C: C.reshape(a.nblock, len(lags), 2, M, 2, M).transpose(0, 2, 4, 3, 5, 1)   # [B, a, b, m, m', lag]
    out = os.path.join(a.run, "dyn_C.npz")
    np.savez(out, Cp=rs(Cp), Cm=rs(Cm), lag_frames=lags, frame_steps=info["cat"][2], L=L,
             n_m=(mean / V).reshape(2, M), m=np.arange(1, M + 1), k=2 * np.pi * np.arange(1, M + 1) / L,
             N=np.array([info["cat"][1], info["ani"][1]]), frames=T, nblock=a.nblock, first_step=info["cat"][3])
    print(f"{out}: {T} frames x {info['cat'][2]} steps, L={L:.4f}, N={info['cat'][1]}+{info['ani'][1]}, "
          f"{len(lags)} lags to {lmax}, {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fkt")
    f.add_argument("--zf", required=True)
    f.add_argument("--mmax", type=int, default=20)
    f.add_argument("--lmax", type=int, default=5000)
    f.add_argument("--nblock", type=int, default=8)
    c = sub.add_parser("cmm")
    c.add_argument("--run", required=True)
    c.add_argument("--mmax", type=int, default=20)
    c.add_argument("--lmax", type=int, default=5000)
    c.add_argument("--nblock", type=int, default=8)
    a = ap.parse_args()
    cmd_fkt(a) if a.cmd == "fkt" else cmd_cmm(a)


if __name__ == "__main__":
    main()
