#!/usr/bin/env python3
"""Aperiodic two-dimensional potentials V_alpha(x, y) = V_N + s_alpha psi (uniform along z): no lattice, irregular
Gaussian features at random positions with random widths (anisotropic allowed), plus a few oblique plane waves.
Same representation as make_potential2d.py (sum of products amp X(x) Y(y) of cosine series; each Gaussian is the
product of two period-L trains, so the potential is still box periodic, as it must be), same LAMMPS include.

  p54 (both): V_N = 6 irregular Gaussian wells and barriers + 2 oblique plane waves;  psi = 3 charged Gaussian wells
  p55 (both): V_N = 3 oblique plane waves + 3 Gaussian barriers;  psi = 4 irregular charged Gaussian wells
Amplitudes inside the training range: pp(V_N) <= 4.5 kT, pp(psi) <= 3 kT.
    make_potential2d_aperiodic.py --lz 24.416564 --outdir <state>/potentials --seed 54
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_potential2d as P2

P2.TOL = 1e-8          # shorter cosine series for the single-Gaussian trains (relative truncation of the far tail)


def irregular2d(rng, L, n, amps, widths, min_gap):
    while True:
        pts = rng.uniform(0, L, (n, 2))
        d = np.abs(pts[:, None, :] - pts[None, :, :]); d = np.minimum(d, L - d); dist = np.sqrt((d ** 2).sum(-1)) + np.eye(n) * 1e9
        if dist.min() >= min_gap:
            break
    out = []
    for (x0, y0) in pts:
        a = float(rng.choice(amps) * np.exp(rng.uniform(-0.3, 0.3)))
        wx = float(np.exp(rng.uniform(np.log(widths[0]), np.log(widths[1])))); wy = float(wx * np.exp(rng.uniform(-0.35, 0.35)))
        out.append((float(x0), float(y0), a, wx, wy))
    return out


def gauss_terms(channel, feats):
    return [dict(channel=channel, amp=a, X=P2.train(L_, 1, wx, x0), Y=P2.train(L_, 1, wy, y0), feature=dict(x=x0, y=y0, amp_kT=a, wx=wx, wy=wy))
            for x0, y0, a, wx, wy in feats]


def wave_terms(channel, rng, pairs, amp):
    return [dict(channel=channel, amp=float(amp * np.exp(rng.uniform(-0.2, 0.2))), X=P2.cosine(L_, mx, float(rng.uniform(0, 2 * np.pi))),
                 Y=P2.cosine(L_, my, float(rng.uniform(0, 2 * np.pi)))) for mx, my in pairs]


def scale_to(terms, channel, pp_max, X, Y):
    v = P2.part(terms, channel, X, Y); f = min(1.0, pp_max / max(np.ptp(v), 1e-12))
    for t in terms:
        if t["channel"] == channel:
            t["amp"] *= f
            if "feature" in t: t["feature"]["amp_kT"] *= f
    return f


def main():
    global L_
    ap = argparse.ArgumentParser()
    ap.add_argument("--lz", type=float, required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=54); ap.add_argument("--temp", type=float, default=1.0)
    a = ap.parse_args(); L_ = a.lz; rng = np.random.default_rng(a.seed)
    x = np.linspace(0, L_, 601); X, Y = np.meshgrid(x, x, indexing="ij")
    pots = {}
    fN = irregular2d(rng, L_, 6, amps=[-2.5, -1.5, 1.5, -2.0, 1.0, -1.2], widths=(0.7, 1.4), min_gap=4.0)
    fZ = irregular2d(rng, L_, 3, amps=[-1.5, -1.2, -1.8], widths=(0.8, 1.2), min_gap=5.0)
    terms = gauss_terms("N", fN) + wave_terms("N", rng, [(1, 2), (2, -1)], 0.35) + gauss_terms("Z", fZ)
    sN, sZ = scale_to(terms, "N", 4.5, X, Y), scale_to(terms, "Z", 3.0, X, Y)
    pots["p54"] = ("2d_aperiodic", "both", terms, f"V_N: 6 irregular Gaussian features (scale {sN:.2f}) + oblique waves (1,2), (2,-1); psi: 3 charged Gaussian wells (scale {sZ:.2f})")
    fN = irregular2d(rng, L_, 3, amps=[1.5, 1.2, 2.0], widths=(0.8, 1.3), min_gap=5.0)
    fZ = irregular2d(rng, L_, 4, amps=[-2.0, -1.5, -2.4, -1.2], widths=(0.7, 1.2), min_gap=4.5)
    terms = wave_terms("N", rng, [(2, 3), (3, -1), (1, 4)], 0.5) + gauss_terms("N", fN) + gauss_terms("Z", fZ)
    sN, sZ = scale_to(terms, "N", 4.5, X, Y), scale_to(terms, "Z", 3.0, X, Y)
    pots["p55"] = ("2d_aperiodic", "both", terms, f"V_N: oblique waves (2,3), (3,-1), (1,4) + 3 Gaussian barriers (scale {sN:.2f}); psi: 4 irregular charged Gaussian wells (scale {sZ:.2f})")
    os.makedirs(a.outdir, exist_ok=True)
    for tag, (fam, kind, terms, note) in pots.items():
        j = os.path.join(a.outdir, tag + ".json")
        if os.path.exists(j): sys.exit(f"{j} exists: refusing to overwrite")
        vn, ps = P2.part(terms, "N", X, Y), P2.part(terms, "Z", X, Y)
        spec = dict(tag=tag, seed=a.seed, kind=kind, family=fam, dim=2, lz=a.lz, temp=a.temp, pp_neutral_kT=float(np.ptp(vn)), pp_charged_kT=float(np.ptp(ps)),
                    terms=[{k: v for k, v in t.items() if k != "feature"} for t in terms], features=[t["feature"] | {"channel": t["channel"]} for t in terms if "feature" in t], note=note)
        json.dump(spec, open(j, "w"), indent=1)
        with open(os.path.join(a.outdir, tag + ".lmp"), "w") as f:
            f.write(f"# two-dimensional external potential '{tag}' ({fam}, {kind}): {note}\n# L = {a.lz:.10g}\n")
            for sp, sign in (("cat", 1.0), ("ani", -1.0)):
                for what in ("fx", "fy", "e"):
                    f.write(f'variable {what}_{sp} atom "{P2.lmp_expr(terms, sign, what)}"\n')
        fmax = max(np.abs(sum(t["amp"] * (1 if t["channel"] == "N" else s) * P2.dseries(t["X"], X) * P2.series(t["Y"], Y) for t in terms)).max() for s in (1.0, -1.0))
        nterm = sum(len(t["X"]) + len(t["Y"]) for t in terms)
        print(f"[{tag}] {kind:5s} pp(V_N) {np.ptp(vn):.2f}  pp(psi) {np.ptp(ps):.2f}  V_+ [{(vn + ps).min():+.2f}, {(vn + ps).max():+.2f}]  V_- [{(vn - ps).min():+.2f}, {(vn - ps).max():+.2f}] kT  "
              f"max|F_x| {fmax:.2f}  {len(terms)} product terms, {nterm} cosines, line {max(len(P2.lmp_expr(terms, 1.0, w)) for w in ('fx', 'fy', 'e'))} chars")


if __name__ == "__main__":
    main()
