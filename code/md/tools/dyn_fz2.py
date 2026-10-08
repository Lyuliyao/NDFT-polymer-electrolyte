#!/usr/bin/env python3
"""<f_z^2> per species from a zero-field ion dump (for the half-step temperature).
    dyn_fz2.py <zero_field dir>  -> <dir>/fz2.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio
d = sys.argv[1]
fn = os.path.join(d, "ions.dump")
steps, boxes, cols, data = dumpio.read_dump(fn, cache=os.path.exists(fn + ".npz"), stride=10)
typ = data[0, :, cols.index("type")].astype(int)
fz = data[:, :, cols.index("fz")]
out = {"cation": float(np.mean(fz[:, typ == 2] ** 2)), "anion": float(np.mean(fz[:, typ == 3] ** 2)),
       "frames_used": int(len(steps)), "stride": 10}
json.dump(out, open(os.path.join(d, "fz2.json"), "w"), indent=1)
print(d, out)
