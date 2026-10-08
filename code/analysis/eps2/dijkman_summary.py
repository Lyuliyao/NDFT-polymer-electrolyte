"""Architecture x training signal: the paper's functional (learned kernels) and the global CNN free energy of
Dijkman et al. (PRL 134, 056103, 2025), each trained by force matching (Eq. 7) and by pair-correlation matching
(zero-field S_ab(k)), three seeds, both eps_r.  Metrics: held-out force chi2 (eps_r = 2 without p27) and profile
error, c = 0.05, Euler-Lagrange convergence, S(k) errors and stability (from metrics.json), Gamma(k_1) on the box
grid against MD, net internal force and reflection error on the MD profiles.
    python figs/dijkman_summary.py      (cwd = <eps2 root>/code)"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
R = "/mnt/research/MultiscaleML_group/Liyao"
sys.path.insert(0, f"{R}/salt_in_polymer_eps2/code")
import learn.data as D
from learn.protocol import load_model
from learn.models import mu_theta, force_density, W_of_k
from learn.evaluate import S_from_W

T = "c001_c002_c004_c006_c008"; V2 = f"joint_v2_L1_C4_R128x128_H64_{T}_val3"
CONCS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08]
l2 = lambda r: 100 * 0.5 * (r["profile"]["cation"]["rel_l2"] + r["profile"]["anion"]["rel_l2"])


def models(e):
    tag = "_long" if e == "7.5" else ""
    KB = "_kbK2"           # 2026-10-04: the force-matched models carry the spectrum loss (p = 2); the pcm rows do not
    return {"ours, force": [f"{V2}_kn1softplus{KB}{tag}_s{s}" for s in (0, 1, 2)],
            "ours, pcm": [f"{V2}_kn1softplus_pcm{tag}_s{s}" for s in (0, 1, 2)],
            "CNN, force": [f"joint_cnn_d2_L1_C4_{T}_val3_force{KB}{tag}_s{s}" for s in (0, 1, 2)],
            "CNN, pcm": [f"joint_cnn_d2_L1_C4_{T}_val3_pcm{tag}_s{s}" for s in (0, 1, 2)],
            "CNN, force, 20k steps": [f"joint_cnn_d2_L1_C4_{T}_val3_force_n20000{KB}{tag}_s0"],
            "CNN, force, shifted and mirrored copies": [f"joint_cnn_d2_L1_C4_{T}_val3_force_aug7R{KB}{tag}_s0"]}


res = {}
for e, root in (("7.5", f"{R}/salt_in_polymer_eps75"), ("2", f"{R}/salt_in_polymer_eps2")):
    L = f"{root}/runs/learn"
    sps, runs = D.load_all(root=root)
    geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in CONCS}
    passed = [r for r in runs if r.status == "PASS" and any(abs(r.conc - c) < 1e-9 for c in CONCS)]
    gm = json.load(open("/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn/interpretation/gamma_md.json" if e == "7.5"
                        else f"{L}/interpretation/gamma_md.json"))
    for lab, names in models(e).items():
        out = []
        for nm in names:
            d = f"{L}/{nm}"
            if not os.path.exists(f"{d}/predict_c005/metrics.json"):
                continue
            m = json.load(open(f"{d}/metrics.json")); p = json.load(open(f"{d}/predict_c005/metrics.json"))
            cfg, ps = load_model(d)
            ho = [r for r in m["heldout_rows"] if not (e == "2" and r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9)]
            s = dict(ho_chi2=np.mean([r["chi2"] for r in ho]), ho_L2=np.mean([l2(r) for r in ho]),
                     c5_chi2=p["transfer"]["chi2"], c5_L2=100 * 0.5 * (p["transfer"]["cation_rel_l2"] + p["transfer"]["anion_rel_l2"]),
                     el=sum(r["el"]["converged"] for r in m["heldout_rows"] + p["transfer_rows"]),
                     nel=len(m["heldout_rows"]) + len(p["transfer_rows"]),
                     params=sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(ps)), steps=m.get("steps_run", 0))
            st = m.get("structure", {})
            s["Sk_ZZ"] = 100 * max(st[c]["rel_rms_ZZ"] for c in st if float(c) in (0.02, 0.04, 0.06, 0.08))
            s["Sk_NN"] = 100 * max(st[c]["rel_rms_NN"] for c in st if float(c) in (0.02, 0.04, 0.06, 0.08))
            s["unstable"] = sum(int(st[c]["n_unstable"]) for c in st)
            dev = []
            for c in CONCS:
                g = geoms[c]; W = np.asarray(W_of_k(ps, cfg, g)); k = np.asarray(g.k)
                Gam = float(1 / (S_from_W(W, k, g.nbar, g.lB)["NN"][1] / (2 * g.nbar)))
                md, er = gm[f"{c:g}"]["shell1"]["Gamma"], gm[f"{c:g}"]["shell1"]["Gamma_err"]
                dev.append((Gam - md) / er)
            s["Gamma_dev"] = dev
            fd = {c: jax.jit(lambda n, g=g: force_density(ps, cfg, n, g)) for c, g in geoms.items()}
            mt = {c: jax.jit(lambda n, g=g: mu_theta(ps, cfg, n, g)) for c, g in geoms.items()}
            noe, refl = [], []
            for r in passed:
                c = round(r.conc, 2); n = jnp.asarray(r.n)
                f = np.asarray(fd[c](n)); noe.append(abs(f.sum()) / np.abs(f).sum())
                mu = np.asarray(mt[c](n)); muR = np.asarray(mt[c](n[:, ::-1]))[:, ::-1]
                refl.append(np.linalg.norm(muR - mu) / np.linalg.norm(mu - mu.mean(axis=1, keepdims=True)))
            s.update(noether_max=float(np.max(noe)), noether_med=float(np.median(noe)), refl_max=float(np.max(refl)), refl_med=float(np.median(refl)),
                     noether_all=[float(x) for x in noe], refl_all=[float(x) for x in refl],
                     p27=[r["chi2"] for r in m["heldout_rows"] if r["tag"] == "p27" and abs(r["conc"] - 0.04) < 1e-9])
            out.append(s)
            print(f"  {e} {lab} {nm[-6:]}: ho chi2 {s['ho_chi2']:.2f} L2 {s['ho_L2']:.2f}% c5 {s['c5_chi2']:.2f}/{s['c5_L2']:.2f}% EL {s['el']}/{s['nel']} "
                  f"Sk {s['Sk_ZZ']:.1f}/{s['Sk_NN']:.1f}% unst {s['unstable']} Gamma dev {np.round(dev, 1).tolist()} net force {s['noether_max']:.1e} refl {s['refl_max']:.1e}", flush=True)
        if out:
            res.setdefault(e, {})[lab] = out
json.dump(res, open(f"{R}/salt_in_polymer_eps2/runs/learn/interpretation/dijkman_summary.json", "w"), indent=1, default=float)
fm = lambda v, f=".2f": f"{np.mean(v):{f}}" + (f"±{np.std(v, ddof=1):{f}}" if len(v) > 1 else "")
print("\neps  model                       held-out chi2   profile%     c=0.05 chi2/profile%    EL       S_ZZ/S_NN max%  unstable  |Gamma dev| max (sigma)  net force max   reflection max")
for e in res:
    for lab, ss in res[e].items():
        print(f"{e:4s} {lab:26s} {fm([s['ho_chi2'] for s in ss]):>14s} {fm([s['ho_L2'] for s in ss]):>11s}   "
              f"{fm([s['c5_chi2'] for s in ss]):>11s} / {fm([s['c5_L2'] for s in ss]):>10s}  {sum(s['el'] for s in ss):4d}/{sum(s['nel'] for s in ss):<4d} "
              f"{np.mean([s['Sk_ZZ'] for s in ss]):5.1f}/{np.mean([s['Sk_NN'] for s in ss]):5.1f}   {sum(s['unstable'] for s in ss):5d}   "
              f"{np.max([np.max(np.abs(s['Gamma_dev'])) for s in ss]):8.1f}   {max(s['noether_max'] for s in ss):10.1e}   {max(s['refl_max'] for s in ss):10.1e}")
