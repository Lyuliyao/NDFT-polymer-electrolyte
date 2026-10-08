"""Pair closure (V1, exact weighted least squares) fitted on the current data set (long-wavelength runs included) at one,
three, four and five concentrations, eps_r = 7.5; held-out chi2 on the held-out runs of the fitted concentrations and,
for the single-concentration fit, the Euler-Lagrange profile errors. Numbers for Results 'Pair closures miss the
concentration dependence' (replaces the older fits made without the long-wavelength runs). Nothing is written to runs/."""
from figstyle import *
from learn import data as D
from learn.train import fit_v1, residual_pulls
from learn.evaluate import el_solve, profile_metrics
from learn.protocol import load_model

sps, runs = D.load_all(); ho = D.heldout_tags(runs)
cfg, _ = load_model(system("7.5")["pair"])      # 2026-10-04: the refitted pair closure lives in E75ROOT
TRAIN_C = [0.01, 0.02, 0.04, 0.06, 0.08]
pool = [r for r in runs if r.status == "PASS" and r.conc in TRAIN_C and r.tag not in ho.get(r.conc, [])]
test = [r for r in runs if r.status == "PASS" and r.conc in TRAIN_C and r.tag in ho.get(r.conc, [])]
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in TRAIN_C}
out = {}
for concs in ([0.04], [0.01, 0.02, 0.04], [0.01, 0.02, 0.04, 0.06], TRAIN_C):
    tr = [r for r in pool if r.conc in concs]
    params, chi_tr, cond = fit_v1(cfg, [b for b, _ in D.make_batches(tr, sps, kboost=2.0, kboost_mode="k").values()])   # the paper's loss (spectrum term, 2026-10-04)
    per = {}
    for c in concs:
        rs = [r for r in test if r.conc == c]; b, _ = D.make_batches(rs, sps)[c]
        p = np.asarray(residual_pulls(params, cfg, b)); per[f"{c:g}"] = float((p ** 2).reshape(len(rs), 2, -1).mean(axis=2).mean(axis=1).mean())
    key = "+".join(f"{c:g}" for c in concs)
    out[key] = {"n_train": len(tr), "train_chi2": float(chi_tr), "heldout_chi2": float(np.mean([v for v in per.values()])), "per_conc": per}
    if concs == [0.04]:
        l2 = []
        for r in [r for r in test if r.conc == 0.04]:
            n = el_solve(params, cfg, geoms[0.04], r.V)[0]; pm = profile_metrics(n, r.n); l2.append((pm["cation"]["rel_l2"], pm["anion"]["rel_l2"]))
        out[key]["profile_l2_cat_an"] = np.mean(l2, axis=0).tolist()
    print(key, json.dumps(out[key]))
json.dump(out, open(os.path.join(ROOT, "paper/figures/pair_closure_sequence.json"), "w"), indent=1)
