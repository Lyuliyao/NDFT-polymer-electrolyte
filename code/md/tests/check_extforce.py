#!/usr/bin/env python3

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import dumpio

spec = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'extchk.json'))
_, _, cols, data = dumpio.read_dump('extcheck.dump', cache=False)
d = data[-1]
typ = d[:, cols.index('type')].astype(int)
z = d[:, cols.index('z')]
fz = d[:, cols.index('fz')]
fx, fy = d[:, cols.index('fx')], d[:, cols.index('fy')]


def F(sign, z):
    o = np.zeros_like(z)
    for t in spec['neutral']:
        o += t['A'] * t['k'] * np.sin(t['k'] * z + t['phase'])
    for t in spec['charged']:
        o += sign * t['A'] * t['k'] * np.sin(t['k'] * z + t['phase'])
    return o


ana = np.where(typ == 2, F(+1.0, z), F(-1.0, z))
rng = np.ptp(ana)
err = np.max(np.abs(fz - ana))


rel = err / rng
ok = rel < 1e-7 and np.abs(fx).max() == 0.0 and np.abs(fy).max() == 0.0
print(f"atoms checked  : {len(z)} ({(typ==2).sum()} cations, {(typ==3).sum()} anions)")
print(f"max |fz - (-dV/dz)| = {err:.3e}  =  {rel:.2e} of the force range {rng:.3f}")
print(f"max |fx|, |fy|      = {np.abs(fx).max():.1e}, {np.abs(fy).max():.1e}   (must be 0)")
print("EXTERNAL FORCE OK" if ok else "*** EXTERNAL FORCE MISMATCH ***")
sys.exit(0 if ok else 1)
