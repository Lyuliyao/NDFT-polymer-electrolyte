"""Spec §7 checks on synthetic data.  Run:  python -m learn.synthetic

 (i)   profiles generated from a known V1 kernel with the Euler-Lagrange
       solver; c_{ab m} recovered by least squares,
 (ii)  delta F / delta n against central finite differences, every variant,
 (iii) the model's S_ZZ obeys Stillinger-Lovett to 1e-6 after extrapolation,
 plus translation equivariance, Hessian symmetry, and V1's analytic W(k).
"""
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from . import data as D
from .geometry import make_geometry, basis_k
from .models import (ModelConfig, init_params, F_ex, mu_theta, force_density,
                     W_of_k, avni_uk)
from .train import fit_v1, v1_design
from .evaluate import el_solve, stillinger_lovett, S_from_W


def check(name, val, tol):
    ok = val < tol
    print(f"  [{'ok' if ok else 'FAIL'}] {name:58s} {val:.2e}  (tol {tol:g})")
    return ok


def random_density(key, geom, contrast=3.0):
    N = geom.z.shape[0]
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (2, 6)) * 0.3
    ph = jax.random.uniform(k2, (2, 6)) * 2 * np.pi
    m = jnp.arange(1, 7)
    u = jnp.sum(a[:, :, None] * jnp.cos(2 * np.pi * m[None, :, None] * geom.z[None, None, :] / geom.L + ph[:, :, None]), axis=1)
    n = jnp.exp(u)
    return geom.nbar * n / jnp.mean(n, axis=1, keepdims=True)


def main():
    ok = True
    L, N, npair, lB = 24.502285, 245, 480, 7.957187947537492
    geom = make_geometry(L, N, npair, lB, 0.04)
    key = jax.random.PRNGKey(1)
    n = random_density(key, geom)

    print("(ii) functional derivative vs central finite differences")
    cfgs = [ModelConfig("v1"), ModelConfig("avni"), ModelConfig("v3", depth=1),
            ModelConfig("v3", depth=2, C=4), ModelConfig("v3", depth=3, C=8),
            ModelConfig("v2", depth=2, C=4), ModelConfig("v3", depth=2, C=4, grad_invariants=True)]
    for cfg in cfgs:
        ps = init_params(cfg, jax.random.PRNGKey(3))
        if cfg.has_pair:
            ps["pair"] = jax.random.normal(jax.random.PRNGKey(5), (3, cfg.M)) * 2.0 / cfg.pair_scale
        if cfg.has_net:   # make the readout non-trivial
            ps["net"]["readout"][-1]["W"] = ps["net"]["readout"][-1]["W"] * 100.0
        mu = np.asarray(mu_theta(ps, cfg, n, geom))
        F = lambda x: float(F_ex(ps, cfg, x, geom))
        eps = 1e-5
        errs = []
        for a, i in [(0, 10), (1, 77), (0, 200), (1, 244)]:
            e = jnp.zeros((2, N)).at[a, i].set(eps)
            fd = (F(n + e) - F(n - e)) / (2 * eps * geom.A * geom.dz)
            errs.append(abs(fd - mu[a, i]) / (np.abs(mu).max() + 1e-30))
        ok &= check(f"dF/dn FD  {cfg.variant} L={cfg.depth} C={cfg.C} gi={cfg.grad_invariants}", max(errs), 1e-6)

        # translation equivariance: shift the input by 7 bins
        f0 = np.asarray(force_density(ps, cfg, n, geom))
        f7 = np.asarray(force_density(ps, cfg, jnp.roll(n, 7, axis=1), geom))
        ok &= check(f"translation equivariance {cfg.variant} L={cfg.depth}",
                    np.abs(np.roll(f0, 7, axis=1) - f7).max() / np.abs(f0).max(), 1e-10)
        W = np.asarray(W_of_k(ps, cfg, geom))
        ok &= check(f"Hessian symmetry W+- = W-+ {cfg.variant} L={cfg.depth}",
                    np.abs(W[:, 0, 1] - W[:, 1, 0]).max() / (np.abs(W).max() + 1e-30), 1e-10)
        if cfg.variant == "v1":
            Bk = basis_k(np.asarray(geom.k), cfg.widths())
            c = cfg.pair_scale * np.asarray(ps["pair"])
            Wan = np.stack([np.stack([c[0] @ Bk, c[2] @ Bk]), np.stack([c[2] @ Bk, c[1] @ Bk])]).transpose(2, 0, 1)
            ok &= check("V1 W(k) jvp vs analytic sum_m c_m exp(-k^2 s_m^2/2)",
                        np.abs(W - Wan).max() / np.abs(Wan).max(), 1e-10)
        if cfg.variant == "avni":
            Wan = np.asarray(avni_uk(geom.k, lB, cfg.avni_a))
            ok &= check("Avni W(k) jvp vs analytic -4 pi lB (1-cos ka)/k^2",
                        np.abs(W[:, 0, 0] - Wan).max() / np.abs(Wan).max(), 1e-10)

    print("(iii) Stillinger-Lovett: extrapolated A of S_ZZ/(2n) / (k^2/kappa_D^2)")
    for cfg in [ModelConfig("pb"), ModelConfig("avni"), ModelConfig("v1"), ModelConfig("v3", depth=2, C=4)]:
        ps = init_params(cfg, jax.random.PRNGKey(3))
        if cfg.has_pair:
            ps["pair"] = jax.random.normal(jax.random.PRNGKey(5), (3, cfg.M)) * 2.0 / cfg.pair_scale
        if cfg.has_net:
            ps["net"]["readout"][-1]["W"] = ps["net"]["readout"][-1]["W"] * 100.0
        A, _ = stillinger_lovett(ps, cfg, geom)
        ok &= check(f"SL  A - 1  {cfg.variant} L={cfg.depth}", abs(A - 1.0), 1e-6)

    print("(i) recover a known V1 kernel from Euler-Lagrange profiles")
    cfg = ModelConfig("v1")
    rng = np.random.default_rng(0)
    c_true = rng.normal(size=(3, cfg.M)) * np.array([[3.0], [3.0], [-3.0]]) * np.geomspace(1, 0.2, cfg.M)[None, :]
    ps_true = {"pair": jnp.asarray(c_true) / cfg.pair_scale}
    # stability of the chosen kernel at nbar
    W = np.asarray(W_of_k(ps_true, cfg, geom))
    lam = np.linalg.eigvalsh(W + np.eye(2)[None] / geom.nbar).min()
    print(f"    kernel stability: min eigenvalue of nbar^-1 + W(k) = {lam:.3f} (must be > 0)")
    # external potentials from the real c=0.04 schedule
    sps = D.discover_state_points()
    sp = sps[0.04]
    pots = sorted(os.listdir(os.path.join(sp.path, "potentials")))
    Vs = []
    for p in [x for x in pots if x.endswith(".json")][:6]:
        spec = json.load(open(os.path.join(sp.path, "potentials", p)))
        z = np.asarray(geom.z)
        vn = sum(t["A"] * np.cos(t["k"] * z + t["phase"]) for t in spec["neutral"]) if spec["neutral"] else 0 * z
        vc = sum(t["A"] * np.cos(t["k"] * z + t["phase"]) for t in spec["charged"]) if spec["charged"] else 0 * z
        Vs.append(np.stack([vn + vc, vn - vc]))
    from .data import Batch
    ns, fs = [], []
    t0 = time.time()
    for V in Vs:
        n_el, info = el_solve(ps_true, cfg, geom, V, mixing=0.1)
        assert info["converged"], info
        ns.append(n_el)
        fs.append(np.asarray(force_density(ps_true, cfg, jnp.asarray(n_el), geom)))
    print(f"    {len(Vs)} EL solutions in {time.time()-t0:.1f}s, last: {info}")
    ns, fs = np.array(ns), np.array(fs)
    noise = 1e-3 * np.sqrt(np.mean(fs ** 2))
    fs_noisy = fs + rng.normal(size=fs.shape) * noise
    w = np.full_like(fs, 1.0 / noise ** 2)
    b = Batch(n=jnp.asarray(ns), f=jnp.asarray(fs_noisy), w=jnp.asarray(w), V=jnp.asarray(np.array(Vs)), geom=geom)
    # design-matrix consistency with the autodiff force
    y, X, _ = v1_design(cfg, b)
    f_lin = X @ c_true.reshape(-1)
    f_ad = (fs - np.array([np.asarray(force_density({"pair": jnp.zeros((3, cfg.M))}, cfg, jnp.asarray(nn), geom)) for nn in ns])).reshape(-1)
    ok &= check("V1 design matrix == autodiff force", np.abs(f_lin - f_ad).max() / np.abs(f_ad).max(), 1e-10)
    ps_fit, chi2, cond = fit_v1(cfg, [b])
    c_fit = cfg.pair_scale * np.asarray(ps_fit["pair"])
    print(f"    chi2/bin of the fit = {chi2:.3f} (noise level 1), design condition number {cond:.1e}")
    Bk = basis_k(np.asarray(geom.k), cfg.widths())
    sig = np.asarray(geom.k) <= 2 * np.pi * 12 / geom.L          # modes the profiles contain
    ok &= check("V1 recovery with 1e-3 noise: u_ab(k) rel. error, k <= k_12",
                np.abs((c_fit - c_true) @ Bk[:, sig]).max() / np.abs(c_true @ Bk[:, sig]).max(), 5e-2)
    print(f"    coefficient recovery max |dc|/|c| = {np.abs(c_fit - c_true).max()/np.abs(c_true).max():.2e} "
          f"(ill-conditioned by design: the widths overlap)")
    b0 = Batch(n=jnp.asarray(ns), f=jnp.asarray(fs), w=jnp.asarray(w), V=b.V, geom=geom)
    ps0, chi20, _ = fit_v1(cfg, [b0])
    ok &= check("V1 recovery without noise: coefficients", np.abs(cfg.pair_scale * np.asarray(ps0["pair"]) - c_true).max() / np.abs(c_true).max(), 1e-6)

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
