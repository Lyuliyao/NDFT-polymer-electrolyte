#!/usr/bin/env python3
"""Collect the zero-field Gamma(k_1) = 1/S(k_1) of every LJ state point into
runs/learn/interpretation/gamma_md.json, the file the summaries read."""
import glob, json, os
R = "/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj"
out = {}
for f in sorted(glob.glob(f"{R}/runs/lj_T1.5_rho*/zero_field/gamma.json")):
    rho = json.load(open(os.path.join(os.path.dirname(f), "D.json")))["conc"]
    out[f"{rho:g}"] = json.load(open(f))
    print(f"rho {rho:g}: Gamma(k1) {out[f'{rho:g}']['shell1']['Gamma']:.3f} +- {out[f'{rho:g}']['shell1']['Gamma_err']:.3f}")
os.makedirs(f"{R}/runs/learn/interpretation", exist_ok=True)
json.dump(out, open(f"{R}/runs/learn/interpretation/gamma_md.json", "w"), indent=1)
