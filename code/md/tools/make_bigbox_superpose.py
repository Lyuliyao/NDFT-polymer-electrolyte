#!/usr/bin/env python3

import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_potential import draw_part, merge_modes, profile, rescale
from make_bigbox_potentials import write
from make_bigbox_aperiodic import gaussians, irregular


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsmall", type=float, required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--outdir", required=True); ap.add_argument("--seed", type=int, default=62)
    ap.add_argument("--pp-long", type=float, default=1.0)
    a = ap.parse_args()
    L = a.n * a.lsmall; merge_modes.lz = L; rng = np.random.default_rng(a.seed)
    k1 = 2 * np.pi / a.lsmall
    between = [m for m in range(a.n, 6 * a.n + 1) if m % a.n]
    trained = [m for m in range(a.n, 6 * a.n + 1) if m % a.n == 0]
    z = np.linspace(0, L, 20001); pp = lambda ms: float(np.ptp(profile(ms, z)))
    longmode = lambda phase: [dict(m=1, A=0.5 * a.pp_long, k=float(2 * np.pi / L), phase=phase)]
    os.makedirs(a.outdir, exist_ok=True)

    feats = irregular(rng, L, 5, amps=[-2.0, -1.2, 1.5, -2.2, 1.0], widths=(0.6, 1.4), min_gap=4.0)
    g, rem = gaussians(L, feats, a.n)
    struct = merge_modes(g + draw_part(rng, L, 1.0, 4, allowed=between))
    if pp(struct) > 4.5 - a.pp_long: struct = rescale(struct, L, 4.5 - a.pp_long)
    vn = merge_modes(longmode(0.3) + struct)
    psi = draw_part(rng, L, 1.5, 3, allowed=trained)
    spec = dict(tag="p62", seed=a.seed, kind="both", family="bigbox_aperiodic", lz=L, temp=1.0,
                pp_neutral_kT=pp(vn), pp_charged_kT=pp(psi), neutral=vn, charged=psi, size_factor=a.n, pp_long_kT=a.pp_long,
                features=[dict(z=z0, amp_kT=am, width=w) for z0, am, w in feats], removed_pp_kT=rem,
                note=f"box mode k_1/{a.n} ({a.pp_long} kT pp, neutral) + irregular Gaussian features (5, {rem:.2f} kT pp of modes below k_1 removed) + modes between the training wavenumbers; psi: modes at training wavenumbers")
    write(spec, a.outdir)

    struct = draw_part(rng, L, 2.0, 4, allowed=between + trained)
    if pp(struct) > 4.5 - a.pp_long: struct = rescale(struct, L, 4.5 - a.pp_long)
    vn = merge_modes(longmode(0.3) + struct)
    feats = irregular(rng, L, 4, amps=[-2.0, -1.5, -2.4, -1.2], widths=(0.7, 1.2), min_gap=5.0)
    g, rem = gaussians(L, feats, a.n)
    gs = merge_modes(g)
    if pp(gs) > 3.0 - a.pp_long: gs = rescale(gs, L, 3.0 - a.pp_long)
    psi = merge_modes(longmode(1.4) + gs)
    spec = dict(tag="p63", seed=a.seed + 1, kind="both", family="bigbox_aperiodic", lz=L, temp=1.0,
                pp_neutral_kT=pp(vn), pp_charged_kT=pp(psi), neutral=vn, charged=psi, size_factor=a.n, pp_long_kT=a.pp_long,
                features=[dict(z=z0, amp_kT=am, width=w) for z0, am, w in feats], removed_pp_kT=rem,
                note=f"box mode k_1/{a.n} ({a.pp_long} kT pp, neutral and charged) + V_N modes at and between the training wavenumbers; psi: irregular charged Gaussian wells (4, {rem:.2f} kT pp of modes below k_1 removed)")
    write(spec, a.outdir)
    for tag in ("p62", "p63"):
        s = json.load(open(os.path.join(a.outdir, tag + ".json")))
        ms = sorted({t["m"] for t in s["neutral"] + s["charged"] if t["A"] >= 1e-3 * max(x["A"] for x in s["neutral"] + s["charged"])})
        m1 = [t for t in s["neutral"] if t["m"] == 1] + [t for t in s["charged"] if t["m"] == 1]
        print(f"{tag}: pp V_N {s['pp_neutral_kT']:.2f} kT, pp psi {s['pp_charged_kT']:.2f} kT; m = 1 terms {[(round(t['A'], 3)) for t in m1]}; "
              f"{len(s['neutral'])}+{len(s['charged'])} terms; modes m {ms[0]}..{ms[-1]} ({len(ms)}); removed below k_1 from the Gaussians: {s['removed_pp_kT']:.2f} kT pp")


if __name__ == "__main__":
    main()
