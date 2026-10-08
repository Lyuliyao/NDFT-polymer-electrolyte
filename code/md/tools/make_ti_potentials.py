#!/usr/bin/env python3

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_potential import lmp_force_expr, lmp_energy_expr

LOBATTO = [(0.0, 1 / 20), ((1 - (3 / 7) ** 0.5) / 2, 49 / 180), (0.5, 16 / 45), ((1 + (3 / 7) ** 0.5) / 2, 49 / 180), (1.0, 1 / 20)]
INTERIOR = [x for x, _ in LOBATTO[1:-1]]


def write(spec, outdir):
    tag = spec["tag"]
    json.dump(spec, open(os.path.join(outdir, tag + ".json"), "w"), indent=2)
    nt, ch = spec["neutral"], spec["charged"]
    with open(os.path.join(outdir, tag + ".lmp"), "w") as f:
        f.write(f"# thermodynamic integration: {spec['ti']['base']} scaled by lambda = {spec['ti']['lambda']:.6f}\n")
        f.write(f"# L_z={spec['lz']:.10g}  pp(V_N)={spec['pp_neutral_kT']:.3f} kT  pp(psi)={spec['pp_charged_kT']:.3f} kT\n")
        f.write(f'variable fz_cat atom "{lmp_force_expr(nt, ch, +1)}"\n')
        f.write(f'variable fz_ani atom "{lmp_force_expr(nt, ch, -1)}"\n')
        f.write(f'variable ez_cat atom "{lmp_energy_expr(nt, ch, +1)}"\n')
        f.write(f'variable ez_ani atom "{lmp_energy_expr(nt, ch, -1)}"\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--base", nargs="+", required=True); ap.add_argument("--first", type=int, default=70)
    a = ap.parse_args()
    pd = os.path.join(a.state, "potentials"); n = a.first
    for b in a.base:
        base = json.load(open(os.path.join(pd, b + ".json")))
        for lam in INTERIOR:
            s = json.loads(json.dumps(base))
            for key in ("neutral", "charged"):
                for t in s[key]:
                    t["A"] = float(t["A"]) * lam
            if "gauss" in s and isinstance(s["gauss"], dict):
                for k in list(s["gauss"]):
                    if k in ("amp", "amp_kT", "depth", "depth_kT", "amp_well_kT", "amp_barrier_kT") and isinstance(s["gauss"][k], (int, float)):
                        s["gauss"][k] = s["gauss"][k] * lam
            s["pp_neutral_kT"] = base["pp_neutral_kT"] * lam; s["pp_charged_kT"] = base["pp_charged_kT"] * lam
            s["tag"] = f"p{n:02d}"; s["ti"] = dict(base=b, **{"lambda": lam}, rule="gauss-lobatto-5")
            write(s, pd); print(f"{s['tag']}: {b} x {lam:.4f}  pp {s['pp_neutral_kT']:.2f}/{s['pp_charged_kT']:.2f}"); n += 1


if __name__ == "__main__":
    main()
