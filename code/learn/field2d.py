from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .models import ACTIVATIONS, mlp, basis_ft

ESIGN = jnp.asarray([1.0, -1.0])


class Geometry2D(NamedTuple):
    L: float
    dx: float
    kx: jnp.ndarray
    ky: jnp.ndarray
    kk: jnp.ndarray
    nbar: float
    n_pairs: float
    lB: float

    @property
    def N(self):
        return self.kx.shape[0]

    @property
    def dV(self):

        return self.L * self.dx * self.dx


def make_geometry2d(L, N, n_pairs, lB):
    dx = L / N
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)
    kk = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    return Geometry2D(L=float(L), dx=float(dx), kx=jnp.asarray(kx), ky=jnp.asarray(ky), kk=jnp.asarray(kk),
                      nbar=float(n_pairs) / float(L) ** 3, n_pairs=float(n_pairs), lB=float(lB))


def rfft2(f):
    return jnp.fft.rfft2(f, axes=(-2, -1))


def irfft2(F, N):
    return jnp.fft.irfft2(F, s=(N, N), axes=(-2, -1))


def basis_k(params, cfg, g):

    k = g.kk.reshape(-1)
    if cfg.kernel_net:
        Kh = basis_ft(params["knet"], cfg, k)
    else:
        Kh = jnp.exp(-0.5 * (k[None, :] * jnp.asarray(cfg.widths())[:, None]) ** 2)
    return Kh.reshape(-1, *g.kk.shape)


def F_ex(params, cfg, n, g):

    if cfg.variant == "pb":
        return 0.0 * jnp.sum(n)
    if cfg.variant not in ("v1", "v2", "v3") or cfg.depth != 1 or cfg.grad_invariants or cfg.log_inputs or cfg.has_pack:
        raise NotImplementedError(f"field2d: variant {cfg.variant} depth {cfg.depth}")
    N = g.N
    conv = irfft2(rfft2(n)[None] * basis_k(params, cfg, g)[:, None], N)
    F = 0.0
    if cfg.has_pair:
        c = cfg.pair_scale * params["pair"]
        mu_p = jnp.einsum("m,mxy->xy", c[0], conv[:, 0]) + jnp.einsum("m,mxy->xy", c[2], conv[:, 1])
        mu_m = jnp.einsum("m,mxy->xy", c[1], conv[:, 1]) + jnp.einsum("m,mxy->xy", c[2], conv[:, 0])
        F = F + 0.5 * g.dV * jnp.sum(n[0] * mu_p + n[1] * mu_m)
    if cfg.has_net:
        h0 = jnp.transpose(conv, (2, 3, 1, 0)).reshape(N, N, -1) / cfg.n_ref
        phi = mlp(params["net"]["readout"], h0, ACTIVATIONS[cfg.activation])[..., 0]
        F = F + g.dV * cfg.n_ref * jnp.sum(phi)
    return F


def mu_theta(params, cfg, n, g):
    return jax.grad(F_ex, argnums=2)(params, cfg, n, g) / g.dV


def coulomb_phi(n, g):
    k2 = jnp.where(g.kk > 0, g.kk ** 2, 1.0)
    F = 4.0 * jnp.pi * g.lB * rfft2(n[0] - n[1]) / k2
    return irfft2(F.at[0, 0].set(0.0), g.N)


def mu_int(params, cfg, n, g):
    return ESIGN[:, None, None] * coulomb_phi(n, g)[None] + mu_theta(params, cfg, n, g)


def grad2(f, g):

    N = g.N
    F = rfft2(f)
    ikx, iky = 1j * g.kx[:, None], 1j * g.ky[None, :]
    if N % 2 == 0:
        ikx = ikx.at[N // 2, 0].set(0.0); iky = iky.at[0, -1].set(0.0)
    return irfft2(F * ikx, N), irfft2(F * iky, N)


def force_density(params, cfg, n, g):

    gx, gy = grad2(mu_int(params, cfg, n, g), g)
    return -n * gx, -n * gy


def _solver(cfg, m_hist=8):
    def solve(params, g, V, u0, beta, tol, maxiter):
        N = g.N
        kD2 = 8.0 * jnp.pi * g.lB * g.nbar
        kerker = jnp.where(kD2 > 0, g.kk ** 2 / jnp.maximum(g.kk ** 2 + kD2, 1e-300), 1.0)

        def n_of_u(u):
            logw = -u
            logw = logw - jnp.max(logw, axis=(1, 2), keepdims=True)
            w = jnp.exp(logw)
            return g.n_pairs * w / (g.dV * jnp.sum(w, axis=(1, 2), keepdims=True))

        def G(u):
            return V + mu_int(params, cfg, n_of_u(u), g)

        def precond(r):
            rN, rZ = r[0] + r[1], r[0] - r[1]
            rZ = irfft2(rfft2(rZ) * kerker, N)
            r = 0.5 * jnp.stack([rN + rZ, rN - rZ])
            return r - jnp.mean(r, axis=(1, 2), keepdims=True)

        def cond(s):
            return (s[1] < maxiter) & (s[2] > tol) & jnp.isfinite(s[2])

        def body(s):
            u, it, err, dX, dR, r_prev, u_prev = s
            gu = G(u)
            r = precond(gu - u)
            err = jnp.max(jnp.abs(n_of_u(gu) - n_of_u(u))) / g.nbar
            slot = (it - 1) % m_hist
            valid = it > 0
            dX = jnp.where(valid, dX.at[slot].set((u - u_prev).ravel()), dX)
            dR = jnp.where(valid, dR.at[slot].set((r - r_prev).ravel()), dR)
            Am = dR @ dR.T
            reg = 1e-10 * (jnp.trace(Am) / m_hist) + 1e-300
            gamma = jnp.linalg.solve(Am + reg * jnp.eye(m_hist), dR @ r.ravel())
            u_new = u + beta * r - ((dX + beta * dR).T @ gamma).reshape(2, N, N)
            return (u_new, it + 1, err, dX, dR, r, u)

        z2 = jnp.zeros((m_hist, 2 * N * N))
        s0 = (u0, 0, jnp.asarray(1e300), z2, z2, jnp.zeros_like(u0), u0)
        u, it, err, *_ = lax.while_loop(cond, body, s0)
        return n_of_u(u), it, err
    return jax.jit(solve)


_SOLVERS = {}


def el_solve(params, cfg, g, V, mixing=0.1, tol=1e-6, maxiter=5000, retries=(0.05, 0.02)):

    if cfg not in _SOLVERS:
        _SOLVERS[cfg] = _solver(cfg)
    V = jnp.asarray(V)
    for mx in (mixing,) + tuple(retries):
        n, it, err = _SOLVERS[cfg](params, g, V, V, mx, tol, maxiter)
        it, err = int(it), float(err)
        if np.isfinite(err) and err <= tol:
            return np.asarray(n), dict(converged=True, iters=it, err=err, mixing=mx)
    return np.asarray(n), dict(converged=False, iters=it, err=err, mixing=mx)
