"""Alternative training signals for the comparison with earlier neural functionals.

The architecture is unchanged; only the data term of the loss differs.

lmu (local chemical-potential balance, the canonical-ensemble loss of Cheng,
arXiv:2608.13506, Eq. 8): in equilibrium the local chemical potential
    mu_loc,a(z) = ln n_a(z) + V_a(z) + e_a phi[n](z) + mu_theta,a[n](z)
is uniform, so its spatial variance over each run and species is minimised.
It uses the MD density profiles and the external potential only, not forces,
and is unweighted as in the original.

pcm (pair-correlation matching, Dijkman et al., PRL 134, 056103): the bulk
Hessian of the model at the uniform density of each training state point,
through S^-1 = nbar^-1 delta + v_C e e^T + W(k), must reproduce the MD partial
structure factors S_ab(k) of the zero-field runs (Sk_grid.npz: k = 2 pi m / L
on the box grid, block errors).  No field run enters.
"""
import os

import numpy as np
import jax
import jax.numpy as jnp

from .models import mu_int, W_of_k, inverse_structure


# ------------------------------------------------------------------ lmu
def lmu_batch_loss(params, cfg, batch, floor=1e-4):
    """sum over runs, species and bins of (mu_loc - <mu_loc>_z)^2, and the count."""
    def one(n, V):
        mu = jnp.log(jnp.maximum(n, floor * batch.geom.nbar)) + V + mu_int(params, cfg, n, batch.geom)
        return mu - jnp.mean(mu, axis=-1, keepdims=True)
    r = jax.vmap(one)(batch.n, batch.V)
    return jnp.sum(r * r), r.size


def lmu_loss(params, cfg, batches):
    tot, cnt = 0.0, 0
    for b in batches:
        l, s = lmu_batch_loss(params, cfg, b)
        tot, cnt = tot + l, cnt + s
    return tot, cnt


# ------------------------------------------------------------------ pcm
class PCMTarget:
    """MD partial structure factors of one state point on its box grid."""
    def __init__(self, sp, geom, rel_floor=0.01):
        f = os.path.join(sp.zero_field, "Sk_grid.npz")
        z = np.load(f)
        if abs(float(z["L"]) - sp.L) > 1e-6:
            raise ValueError(f"{f}: L {float(z['L'])} != state point {sp.L}")
        self.m = jnp.asarray(z["m"]).astype(int)
        self.k = jnp.asarray(z["k"])
        S = np.stack([z["S_pp"], z["S_mm"], z["S_pm"]])
        E = np.stack([z["S_pp_err"], z["S_mm_err"], z["S_pm_err"]])
        scale = np.stack([z["S_pp"], z["S_mm"], np.sqrt(np.abs(z["S_pp"] * z["S_mm"]))])
        self.S = jnp.asarray(S)
        self.sig = jnp.asarray(np.sqrt(E ** 2 + (rel_floor * scale) ** 2))
        self.geom = geom


def pcm_model_S(params, cfg, t):
    """Model S_pp, S_mm, S_pm at the target's k (3, K)."""
    W = W_of_k(params, cfg, t.geom)[t.m]                                  # (K, 2, 2)
    Si = inverse_structure(W, t.geom, ks=t.k)                            # (K, 2, 2)
    det = Si[:, 0, 0] * Si[:, 1, 1] - Si[:, 0, 1] * Si[:, 1, 0]
    return jnp.stack([Si[:, 1, 1] / det, Si[:, 0, 0] / det, -Si[:, 0, 1] / det])


def pcm_loss(params, cfg, targets):
    tot, cnt = 0.0, 0
    for t in targets:
        r = (pcm_model_S(params, cfg, t) - t.S) / t.sig
        tot, cnt = tot + jnp.sum(r * r), cnt + r.size
    return tot, cnt
