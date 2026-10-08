#!/usr/bin/env python3

import os
import sys

import numpy as np
import math
erfc = np.vectorize(math.erfc)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import model as M

G_EWALD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50055051


def read(fn, keyword):

    sections, cur, inblock = {}, None, False
    for ln in open(fn):
        t = ln.split()
        if len(t) == 1 and t[0].isalpha() or (len(t) == 1 and t[0].isupper()):
            cur = t[0]
            sections.setdefault(cur, []).append([])
            inblock = False
            continue
        if t and t[0] == "N":
            inblock = True
            continue
        if inblock and len(t) == 4 and t[0].isdigit():
            sections[cur][-1].append([float(t[1]), float(t[2]), float(t[3])])
    if keyword not in sections:
        raise SystemExit(f"{fn}: no section '{keyword}' (found {list(sections)})")
    blocks = [b for b in sections[keyword] if b]
    if len(blocks) != 1:
        raise SystemExit(f"{fn}: section '{keyword}' appears {len(blocks)} times "
                         f"-- pair_write appended to a stale file; delete it first")
    a = np.array(blocks[0])
    return a[:, 0], a[:, 1], a[:, 2]


def lj(r, sig, rc, shift=False):
    u = 4.0 * M.EPSILON * ((sig / r) ** 12 - (sig / r) ** 6)
    fo = 24.0 * M.EPSILON * (2.0 * (sig / r) ** 12 - (sig / r) ** 6) / r
    if shift:
        u = u - 4.0 * M.EPSILON * ((sig / rc) ** 12 - (sig / rc) ** 6)
    u = np.where(r < rc, u, 0.0)
    fo = np.where(r < rc, fo, 0.0)
    return u, fo


def solv(r, sig):
    u = -M.S_SOLV * ((sig / r) ** 4 - (sig / M.RC_SOLV) ** 4)
    f = -4.0 * M.S_SOLV * sig ** 4 / r ** 5
    m = r < M.RC_SOLV
    return np.where(m, u, 0.0), np.where(m, f, 0.0)


def coul_real(r, qi, qj, rc):
    u = qi * qj * erfc(G_EWALD * r) / (M.EPS_R * r)
    f = qi * qj / M.EPS_R * (erfc(G_EWALD * r) / r ** 2
                             + 2.0 * G_EWALD / np.sqrt(np.pi) * np.exp(-(G_EWALD * r) ** 2) / r)
    m = r < rc
    return np.where(m, u, 0.0), np.where(m, f, 0.0)


def report(name, r, e, f, ea, fa):

    de = np.max(np.abs(e - ea)) / max(np.max(np.abs(ea)), 1e-9)
    df = np.max(np.abs(f - fa)) / max(np.max(np.abs(fa)), 1e-9)
    ok = "OK " if (de < 1e-5 and df < 1e-5) else "FAIL"
    print(f"  [{ok}] {name:12s} max err / range: energy {de:.2e}  force {df:.2e}"
          f"   (abs: {np.max(np.abs(e-ea)):.2e}, {np.max(np.abs(f-fa)):.2e})")
    return de < 1e-5 and df < 1e-5


print(f"pair_write validation (g_ewald = {G_EWALD})")
allok = True


r, e, f = read("tests/pw_mono_mono.dat", "MONOMONO")
ea, fa = lj(r, 1.0, M.CUT_MM, shift=True)
allok &= report("mono-mono", r, e, f, ea, fa)

SC = M.solv_sigma(M.T_MONO, M.T_CAT)
SA = M.solv_sigma(M.T_MONO, M.T_ANI)
print(f"solvation length: SOLV_SIGMA = {M.SOLV_SIGMA!r} -> cation {SC}, anion {SA}")


r, e, f = read("tests/pw_cat_mono.dat", "CATMONO")
u1, f1 = lj(r, 0.7, M.WCA * 0.7, shift=True)
u2, f2 = solv(r, SC)
allok &= report("cat-mono", r, e, f, u1 + u2, f1 + f2)


r, e, f = read("tests/pw_ani_mono.dat", "ANIMONO")
u1, f1 = lj(r, 1.3, M.WCA * 1.3, shift=True)
u2, f2 = solv(r, SA)
allok &= report("ani-mono", r, e, f, u1 + u2, f1 + f2)


r, e, f = read("tests/pw_cat_ani.dat", "CATANI")
u1, f1 = lj(r, 1.0, M.WCA * 1.0, shift=True)
u2, f2 = coul_real(r, M.QSTAR, -M.QSTAR, 5.0)
allok &= report("cat-ani", r, e, f, u1 + u2, f1 + f2)


print("\nspot checks against the specification:")
print(f"  U_solv(cation-monomer, contact 0.70) = {solv(np.array([0.7]),SC)[0][0]:+.4f} eps")
print(f"  U_solv(anion-monomer,  contact 1.30) = {solv(np.array([1.3]),SA)[0][0]:+.4f} eps")
print(f"  bare Coulomb(+,- at r=1.0 sigma)     = {-M.QSTAR**2/(M.EPS_R*1.0):+.4f} eps  (doc: about -8 kT)")
print(f"  l_B(T*=1.0)                          = {M.bjerrum_sigma(1.0):.3f} sigma  (doc: 8.0)")
print("\nALL PAIR POTENTIALS MATCH" if allok else "\n*** MISMATCH ***")
sys.exit(0 if allok else 1)
