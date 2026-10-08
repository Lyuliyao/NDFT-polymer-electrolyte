"""Gamma = 2 nbar / S_NN(k_1) of saved models, computed exactly as
paper/figures/make_fig3.py does (S from W(k) on an 8x finer grid, Coulomb
included, value at k_1 = 2 pi / L), for the state points of $SIP_ROOT.
    SIP_ROOT=<root> python gamma_k1.py <model dir> [...]   -> JSON on stdout"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.environ["SIP_ROOT"], "code"))
from learn import data as D
from learn.models import W_of_k, fine_geometry
from learn.evaluate import S_from_W
from learn.protocol import load_model

sps = D.discover_state_points()
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in sorted(sps) if c <= 0.08}
out = {}
for d in sys.argv[1:]:
    cfg, ps = load_model(d)
    g = {}
    for c, geo in geoms.items():
        gf = fine_geometry(geo, 8); k = np.asarray(gf.k); k1 = 2 * np.pi / geo.L
        S = S_from_W(np.asarray(W_of_k(ps, cfg, gf)), k, geo.nbar, geo.lB)["NN"] / (2 * geo.nbar)
        g[f"{c:g}"] = float(1.0 / S[np.argmin(abs(k - k1))])
    out[os.path.basename(d.rstrip("/"))] = g
print(json.dumps(out, indent=1))
