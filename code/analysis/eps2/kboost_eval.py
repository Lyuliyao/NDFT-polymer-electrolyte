"""Long-wavelength weighting: the paper's functional (kboost 0) against retrainings with the m = 1, 2 residual of the
long-wavelength runs weighted by (3/m)^kb, kb = 1, 2, 4 (three seeds each, both eps_r), and the wider-kernel variants kn2/kn3 (s_max 2, 3) with and without the weighting.  For every model: held-out chi2 and
profile error (without p27 at eps_r = 2), c = 0.05, Gamma(k_1) at every concentration against MD, and the ratio of
predicted to MD amplitude of the m = 1 number-density mode on the long-wavelength runs of the training box (sum over the
runs of each concentration, all splits).      python figs/kboost_eval.py   (cwd = <eps2 root>/code)"""
import json, os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"; R = "/mnt/research/MultiscaleML_group/Liyao"
sys.path.insert(0, f"{R}/salt_in_polymer_eps2/code")
import learn.data as D
from learn.protocol import load_model
from learn.evaluate import el_solve, S_from_W
from learn.models import W_of_k, fine_geometry
import glob
T = "c001_c002_c004_c006_c008"; V2 = f"joint_v2_L1_C4_R128x128_H64_{T}_val3"; CONCS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]
GMD = json.load(open(f"{S}/paper/figures/fig3_metrics.json"))["Gamma"]
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
amp = lambda y: 2 * abs(np.fft.rfft(y)[1]) / len(y)
res = {}
for e, data, mods, tag in (("7.5", S, f"{R}/salt_in_polymer_eps75/runs/learn", "_long"), ("2", f"{R}/salt_in_polymer_eps2", f"{R}/salt_in_polymer_eps2/runs/learn", "")):
    sps, runs = D.load_all(root=data)
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CONCS}
    long = [r for r in runs if r.status == "PASS" and r.family in ("long", "long3") and r.conc in CONCS]
    cands = [(f"kb{kb}", s, f"{mods}/{V2}_kn1softplus{'' if kb == 0 else f'_kb{kb}'}{tag}_s{s}") for kb in (0, 1, 2, 4) for s in (0, 1, 2)]
    cands += [(f"kbA{kb}", s, f"{mods}/{V2}_kn1softplus_kbA{kb}{tag}_s{s}") for kb in (2, 4) for s in (0, 1, 2)]   # spectrum loss, all modes, all runs
    cands += [(f"kbK{kb}", s, f"{mods}/{V2}_kn1softplus_kbK{kb}{tag}_s{s}") for kb in (2, 4) for s in (0, 1, 2)]   # spectrum loss k_m^-p, no reference
    for d in sorted(glob.glob(f"{mods}/{V2}_kn[23]*softplus*_s?") + glob.glob(f"{mods}/{V2}_kn1softplus_w*_s?")):
        n = os.path.basename(d)
        if any(x in n for x in ("dz005", "pcm", "nostab", "f025", "f050")): continue
        cands.append((n.split("_val3_")[1].replace(tag + "_s", "_s").rsplit("_s", 1)[0], int(n[-1]), d))
    for kb, s, d in cands:
            if not os.path.exists(f"{d}/predict_c005/metrics.json"):
                print(f"eps {e} {kb} s{s}: not finished"); continue
            m = json.load(open(f"{d}/metrics.json")); p = json.load(open(f"{d}/predict_c005/metrics.json")); cfg, ps = load_model(d)
            ho = [r for r in m["heldout_rows"] if not (e == "2" and r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9)]
            st = m["structure"]
            gk1 = {}
            for c in CONCS:
                g = geoms[c]; gf = fine_geometry(g, 8); k = np.asarray(gf.k)
                SNN = S_from_W(np.asarray(W_of_k(ps, cfg, gf)), k, g.nbar, g.lB)["NN"] / (2 * g.nbar)
                gk1[f"{c:g}"] = float(1 / SNN[int(np.argmin(abs(k - 2 * np.pi / g.L)))])
            row = dict(Gamma_k1=gk1, s_max=cfg.s_max, ho_chi2=float(np.mean([r["chi2"] for r in ho])), ho_L2=float(np.mean([l2(r) for r in ho])), c5_chi2=p["transfer"]["chi2"],
                       c5_L2=100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"]),
                       Gamma={c: st[c]["Gamma"] for c in st}, unstable=sum(int(st[c]["n_unstable"]) for c in st), ratio={})
            for c in CONCS:
                rs = [r for r in long if abs(r.conc - c) < 1e-9]
                if not rs: continue
                a_md = a_el = 0.0
                for r in rs:
                    n_el, _ = el_solve(ps, cfg, geoms[c], r.V); a_md += amp(r.n.sum(0)); a_el += amp(np.asarray(n_el).sum(0))
                row["ratio"][f"{c:g}"] = a_el / a_md
            res[f"{e}/kb{kb}/s{s}"] = row
            gd = {c: (row["Gamma_k1"][c] - GMD[e]["MD"][c][0]) / GMD[e]["MD"][c][1] for c in ("0.01", "0.04", "0.08")}
            print(f"eps {e} {kb} s{s} (s_max {cfg.s_max}): held-out chi2 {row['ho_chi2']:.3f} L2 {row['ho_L2']:.2f}% c=0.05 {row['c5_chi2']:.3f}/{row['c5_L2']:.2f}% unstable {row['unstable']}; "
                  f"m=1 ratio by c {dict((k, round(v, 3)) for k, v in row['ratio'].items())}; Gamma dev (sigma) {dict((k, round(v, 1)) for k, v in gd.items())}", flush=True)
json.dump(res, open(f"{R}/salt_in_polymer_eps2/runs/learn/interpretation/kboost_eval.json", "w"), indent=1, default=float)
print("\nsummary (mean over seeds):")
for e in ("7.5", "2"):
    for kb in sorted({k.split("/")[1] for k in res if k.startswith(e + "/")}):
        rows = [res[k] for k in res if k.startswith(f"{e}/{kb}/")]
        if not rows: continue
        rat = {c: np.mean([r["ratio"][c] for r in rows]) for c in rows[0]["ratio"]}
        print(f"  eps {e} {kb} (s_max {rows[0]['s_max']}, {len(rows)} seeds): held-out chi2 {np.mean([r['ho_chi2'] for r in rows]):.3f}, profile {np.mean([r['ho_L2'] for r in rows]):.2f}%, c=0.05 chi2 {np.mean([r['c5_chi2'] for r in rows]):.3f}; "
              f"k1 amplitude ratio {dict((k, round(v, 3)) for k, v in rat.items())}; Gamma(k1) c=0.04 {np.mean([r['Gamma_k1']['0.04'] for r in rows]):.2f} (MD {GMD[e]['MD']['0.04'][0]:.2f}±{GMD[e]['MD']['0.04'][1]:.2f})")
