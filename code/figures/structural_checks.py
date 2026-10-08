"""Structural properties on the TRAINED functionals, evaluated on the MD profiles of every accepted run:
(a) Noether force sum rule  sum_alpha int f_alpha^int dz = 0  (Prop. 3a of the closure note);
(b) Hessian symmetry W_+-(k) = W_-+(k) at every concentration (Prop. 2a);
(c) translation equivariance of the force (Prop. 1, shift by 7 bins);
(d) Stillinger-Lovett intercept (Prop. 4a) at every concentration, via learn.evaluate.stillinger_lovett."""
from figstyle import *
import jax.numpy as jnp
from learn import data as D
from learn.models import W_of_k, force_density, F_ex, mu_theta
from learn.evaluate import stillinger_lovett
from learn.protocol import load_model
sps, runs = D.load_all()
geoms = {c: D.geometry_of(sps[c], int(round(sps[c].L / 0.1))) for c in sorted(sps)}
out = {}
for name, mdl in (("functional", BEST), ("pair-kernel functional", system("7.5")["pair"])):      # 2026-10-04: the refitted pair closure lives in E75ROOT
    cfg, ps = load_model(mdl if os.path.isabs(mdl) else os.path.join(ROOT, "runs/learn", mdl))
    rel_force, rel_eq = [], []
    for r in runs:
        if r.status != "PASS" or r.conc not in CONCS: continue
        g = geoms[r.conc]; n = jnp.asarray(r.n)
        f = np.asarray(force_density(ps, cfg, n, g))                      # (2, N) internal force density incl. Coulomb
        tot = f.sum(axis=0).sum() * g.dz; scale = np.abs(f).sum() * g.dz
        rel_force.append(abs(tot) / scale)
        f7 = np.asarray(force_density(ps, cfg, jnp.roll(n, 7, axis=1), g))
        rel_eq.append(np.abs(np.roll(f, 7, axis=1) - f7).max() / np.abs(f).max())
    wsym, sl = {}, {}
    for c in CONCS:
        g = geoms[c]; W = np.asarray(W_of_k(ps, cfg, g))
        wsym[c] = float(np.abs(W[:, 0, 1] - W[:, 1, 0]).max() / np.abs(W).max())
        A, _ = stillinger_lovett(ps, cfg, g); sl[c] = float(abs(A - 1))
    out[name] = {"n_runs": len(rel_force), "noether_force_max_rel": float(np.max(rel_force)), "noether_force_median_rel": float(np.median(rel_force)),
                 "translation_equivariance_max_rel": float(np.max(rel_eq)), "W_symmetry_max_rel": max(wsym.values()), "SL_intercept_max_abs": max(sl.values()),
                 "SL_by_conc": sl}
    print(name, json.dumps(out[name], indent=1))
json.dump(out, open(os.path.join(ROOT, "paper/figures/structural_checks.json"), "w"), indent=1)
