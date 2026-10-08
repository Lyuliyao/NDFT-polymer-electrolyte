"""Grid-refinement test: the paper's functional trained on profiles binned at 0.1 sigma (paper models) against the
same training on the same runs binned at 0.05 sigma (roots *_dz005, legacy error model), seed by seed.
  bulk:   relative rms difference of W_ab(k) (k <= 2.1) between the two models at every concentration; Gamma(k_1)
          of both against MD; structure-factor errors (metrics.json);  real-space W_--(sigma), W_++(sigma)
  data:   held-out chi2 and profile error of each model on its own grid (metrics.json), c = 0.05 likewise;
  cross:  each model evaluated on the other grid's held-out runs (chi2 on the sampled profiles, EL profile error).
    python figs/grid_compare.py        (cwd = <eps2 root>/code)"""
import json, os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
S = "/mnt/gs21/scratch/lyuliyao/salt_in_polymer"; R = "/mnt/research/MultiscaleML_group/Liyao"
sys.path.insert(0, f"{R}/salt_in_polymer_eps2/code")
import learn.data as D
from learn.protocol import load_model
from learn.models import W_of_k, fine_geometry
from learn.evaluate import S_from_W, heldout_report
from learn.interpret import W_r_from_k
T = "c001_c002_c004_c006_c008"; V2 = f"joint_v2_L1_C4_R128x128_H64_{T}_val3"
CONCS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]
GMD = json.load(open(f"{S}/paper/figures/fig3_metrics.json"))["Gamma"]
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])
SYS = {"7.5": dict(data=S, models=f"{R}/salt_in_polymer_eps75/runs/learn", tag="_long", fine=f"{R}/salt_in_polymer_eps75_dz005"),
       "2": dict(data=f"{R}/salt_in_polymer_eps2", models=f"{R}/salt_in_polymer_eps2/runs/learn", tag="", fine=f"{R}/salt_in_polymer_eps2_dz005")}


def own_metrics(d, e):
    m = json.load(open(f"{d}/metrics.json")); p = json.load(open(f"{d}/predict_c005/metrics.json"))
    ho = [r for r in m["heldout_rows"] if not (e == "2" and r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9)]
    st = m["structure"]
    return dict(ho_chi2=np.mean([r["chi2"] for r in ho]), ho_L2=np.mean([l2(r) for r in ho]), c5_chi2=p["transfer"]["chi2"],
                c5_L2=100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"]),
                sk=np.mean([100 * st[c]["rel_rms_ZZ"] for c in st if float(c) in (0.02, 0.04, 0.05, 0.06, 0.08)]),
                skN=np.mean([100 * st[c]["rel_rms_NN"] for c in st if float(c) in (0.02, 0.04, 0.05, 0.06, 0.08)]),
                unstable=sum(int(st[c]["n_unstable"]) for c in st), dz=round(m.get("dz", 0), 3))


def cross(params, cfg, root, e):
    """held-out runs (without p27) and the c = 0.05 runs of `root`, evaluated with the given model."""
    sps, runs = D.load_all(root=root); ho = D.heldout_tags(runs, root=root)
    hor = [r for r in runs if r.status == "PASS" and r.conc in ho and r.tag in ho[r.conc] and not (e == "2" and r.tag == "p27" and abs(r.conc - 0.04) < 1e-9)]
    c5 = [r for r in runs if r.status == "PASS" and abs(r.conc - 0.05) < 1e-9]
    out = {}
    for lab, rs in (("ho", hor), ("c5", c5)):
        rows = heldout_report(params, cfg, D.make_batches(rs, sps), el=True, log=lambda *a, **k: None)
        out[lab + "_chi2"] = float(np.mean([r["chi2"] for r in rows])); out[lab + "_L2"] = float(np.mean([l2(r) for r in rows]))
        out[lab + "_el"] = f"{sum(r['el']['converged'] for r in rows)}/{len(rows)}"
    N = rs[0].n.shape[1]; out["N_c005"] = int(N)
    return out


res = {}
for e, sy in SYS.items():
    sps, _ = D.load_all(root=sy["data"], concs=CONCS)
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CONCS}
    for s in (0, 1, 2):
        dc, df = f"{sy['models']}/{V2}_kn1softplus_kbK2{sy['tag']}_s{s}", f"{sy['fine']}/runs/learn/{V2}_kn1softplus_dz005_kbK2{sy['tag']}_s{s}"   # 2026-10-04: spectrum loss
        if not os.path.exists(f"{df}/predict_c005/metrics.json"):
            print(f"eps {e} seed {s}: fine model not finished"); continue
        (cc, pc), (cf, pf) = load_model(dc), load_model(df)
        row = dict(coarse=own_metrics(dc, e), fine=own_metrics(df, e), W={}, Gamma={}, Wr={})
        for c in CONCS:
            gf = fine_geometry(geoms[c], 8); k = np.asarray(gf.k); sel = (k > 0) & (k <= 2.1)
            Wc, Wf = np.asarray(W_of_k(pc, cc, gf)), np.asarray(W_of_k(pf, cf, gf))
            row["W"][f"{c:g}"] = {ch: float(np.sqrt(np.mean((Wf[sel, a, b] - Wc[sel, a, b]) ** 2) / np.mean(Wc[sel, a, b] ** 2)))
                                 for ch, (a, b) in (("++", (0, 0)), ("--", (1, 1)), ("+-", (0, 1)))}
            i1 = int(np.argmin(abs(k - 2 * np.pi / geoms[c].L))); g = geoms[c]
            row["Gamma"][f"{c:g}"] = [float(1 / (S_from_W(W, k, g.nbar, g.lB)["NN"][i1] / (2 * g.nbar))) for W in (Wc, Wf)] + GMD[e]["MD"][f"{c:g}"]
            if c in (0.01, 0.08):
                rr = np.array([1.0])
                row["Wr"][f"{c:g}"] = {ch: [float(W_r_from_k(k, W[:, a, b], rr)[0]) for W in (Wc, Wf)] for ch, (a, b) in (("--", (1, 1)), ("++", (0, 0)))}
        row["coarse_on_fine"] = cross(pc, cc, sy["fine"], e); row["fine_on_coarse"] = cross(pf, cf, sy["data"], e)
        res[f"{e}/s{s}"] = row
        dW = max(v for c in row["W"].values() for v in c.values())
        print(f"eps {e} seed {s}: held-out chi2 {row['coarse']['ho_chi2']:.3f} (0.1) vs {row['fine']['ho_chi2']:.3f} (0.05); profile {row['coarse']['ho_L2']:.2f} vs {row['fine']['ho_L2']:.2f}%; "
              f"c=0.05 chi2 {row['coarse']['c5_chi2']:.3f} vs {row['fine']['c5_chi2']:.3f}; S_ZZ/S_NN err {row['coarse']['sk']:.1f}/{row['coarse']['skN']:.1f} vs {row['fine']['sk']:.1f}/{row['fine']['skN']:.1f}%; "
              f"unstable {row['coarse']['unstable']} vs {row['fine']['unstable']}; max rel W(k) difference {100 * dW:.1f}%; "
              f"Gamma(k1) 0.1/0.05/MD at c=0.04: {row['Gamma']['0.04'][0]:.2f}/{row['Gamma']['0.04'][1]:.2f}/{row['Gamma']['0.04'][2]:.2f}; "
              f"W--(sigma) at c=0.01: {row['Wr']['0.01']['--'][0]:.2f} vs {row['Wr']['0.01']['--'][1]:.2f}; "
              f"cross: 0.1-model on 0.05 data chi2 {row['coarse_on_fine']['ho_chi2']:.3f} L2 {row['coarse_on_fine']['ho_L2']:.2f}%, 0.05-model on 0.1 data chi2 {row['fine_on_coarse']['ho_chi2']:.3f} L2 {row['fine_on_coarse']['ho_L2']:.2f}%", flush=True)
json.dump(res, open(f"{R}/salt_in_polymer_eps2/runs/learn/interpretation/grid_compare.json", "w"), indent=1, default=float)
