"""Evaluation (spec §6): held-out residuals, Euler-Lagrange profiles, S(k),
the thermodynamic factor, and the Stillinger-Lovett code check."""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from scipy.interpolate import CubicSpline

from .geometry import E_SIGN, SPECIES, esign, make_geometry, coulomb_vk
from .models import mu_int, W_of_k, lambda_min_2x2, inverse_structure, fine_geometry, sym_W
from .train import chi2_per_bin, residual_pulls


# ------------------------------------------------------ Euler-Lagrange solver
def _el_solver(cfg, m_hist=8):
    """Self-consistent solution of n_a = N_a exp(-u_a)/Z_a, u = V + mu_int[n].

    Mixing is done on the potentials u (any u gives a positive, normalized n),
    with Anderson acceleration over the last `m_hist` steps and a Kerker
    preconditioner k^2/(k^2 + kappa_D^2) on the charge channel.  Plain Picard
    mixing cannot converge here: the mean-field Coulomb term gives the
    lowest box mode of rho_Z a linear gain kappa_D^2/k_1^2 ~ 25-100 at these
    concentrations, which would need a mixing below 0.02."""
    def solve(params, geom, V, u0, beta, tol, maxiter):
        N = geom.z.shape[0]
        Nalpha = geom.n_pairs
        k = geom.k
        kD2 = 8.0 * jnp.pi * geom.lB * geom.nbar
        # no charges (l_B = 0, the LJ benchmark): no screening to precondition
        kerker = jnp.where(kD2 > 0, k * k / jnp.maximum(k * k + kD2, 1e-300), 1.0)

        def n_of_u(u):
            logw = -u
            logw = logw - jnp.max(logw, axis=1, keepdims=True)
            wgt = jnp.exp(logw)
            return Nalpha * wgt / (geom.A * geom.dz * jnp.sum(wgt, axis=1, keepdims=True))

        def G(u):
            return V + mu_int(params, cfg, n_of_u(u), geom)

        nsp = u0.shape[0]

        def precond(r):
            if nsp == 2:                                     # Kerker on the charge channel of the ion pair
                rN, rZ = r[0] + r[1], r[0] - r[1]
                rZ = jnp.fft.irfft(jnp.fft.rfft(rZ) * kerker, n=N)
                r = 0.5 * jnp.stack([rN + rZ, rN - rZ])
            return r - jnp.mean(r, axis=1, keepdims=True)      # the gauge of u

        def cond(s):
            return (s[1] < maxiter) & (s[2] > tol) & jnp.isfinite(s[2])

        def body(s):
            u, it, err, dX, dR, r_prev, u_prev = s
            g = G(u)
            r = precond(g - u)
            err = jnp.max(jnp.abs(n_of_u(g) - n_of_u(u))) / geom.nbar
            slot = (it - 1) % m_hist
            valid = it > 0
            dX = jnp.where(valid, dX.at[slot].set((u - u_prev).ravel()), dX)
            dR = jnp.where(valid, dR.at[slot].set((r - r_prev).ravel()), dR)
            Am = dR @ dR.T
            reg = 1e-10 * (jnp.trace(Am) / m_hist) + 1e-300
            gamma = jnp.linalg.solve(Am + reg * jnp.eye(m_hist), dR @ r.ravel())
            u_new = u + beta * r - ((dX + beta * dR).T @ gamma).reshape(nsp, N)
            return (u_new, it + 1, err, dX, dR, r, u)

        z2 = jnp.zeros((m_hist, nsp * N))
        s0 = (u0, 0, jnp.asarray(1e300), z2, z2, jnp.zeros_like(u0), u0)
        u, it, err, *_ = lax.while_loop(cond, body, s0)
        return n_of_u(u), it, err
    return jax.jit(solve)


_SOLVERS = {}


def el_solve(params, cfg, geom, V, mixing=0.2, tol=1e-6, maxiter=5000, u0=None,
             retries=(0.1, 0.05, 0.02)):
    """Canonical Euler-Lagrange equation, see _el_solver.  Retries with a
    smaller mixing when the iteration does not converge.  Returns (n, info)."""
    if cfg not in _SOLVERS:
        _SOLVERS[cfg] = _el_solver(cfg)
    solve = _SOLVERS[cfg]
    V = jnp.asarray(V)
    u0 = V if u0 is None else jnp.asarray(u0)
    for mx in (mixing,) + tuple(retries):
        n, it, err = solve(params, geom, V, u0, mx, tol, maxiter)
        it, err = int(it), float(err)
        if np.isfinite(err) and err <= tol:
            return np.asarray(n), dict(converged=True, iters=it, err=err, mixing=mx)
    return np.asarray(n), dict(converged=False, iters=it, err=err, mixing=mx)


def profile_metrics(n_pred, n_md):
    """Per species: relative L2 error, relative error at the MD density
    maximum and minimum."""
    out = {}
    for a, lab in enumerate(SPECIES[:n_pred.shape[0]]):
        p, m = n_pred[a], n_md[a]
        imax, imin = int(np.argmax(m)), int(np.argmin(m))
        out[lab] = dict(rel_l2=float(np.linalg.norm(p - m) / np.linalg.norm(m)),
                        err_at_max=float((p[imax] - m[imax]) / m[imax]),
                        err_at_min=float((p[imin] - m[imin]) / m[imin]),
                        contrast_md=float(m.max() / max(m.min(), 1e-12)),
                        contrast_pred=float(p.max() / max(p.min(), 1e-12)))
    return out


# ------------------------------------------------------------ structure
def S_from_W(W, k, nbar, lB):
    """S_ab(k) = [nbar^-1 delta + v_Coul(k) e_a e_b + W(k)]^-1, per volume."""
    W, k = np.asarray(W), np.asarray(k)
    nsp = W.shape[-1]; e = esign(nsp)
    ee = np.outer(e, e)
    Minv = np.eye(nsp)[None] / nbar + coulomb_vk(k, lB)[:, None, None] * ee[None] + W
    S = np.linalg.inv(Minv)
    if nsp == 1:                                    # one species: the number channel only
        return dict(k=k, pp=S[:, 0, 0], NN=S[:, 0, 0])
    off = 0.5 * (S[:, 0, 1] + S[:, 1, 0])          # S_+- = S_-+ unless W is not symmetric (c1win)
    return dict(k=k, pp=S[:, 0, 0], mm=S[:, 1, 1], pm=off,
                ZZ=S[:, 0, 0] + S[:, 1, 1] - 2 * off,
                NN=S[:, 0, 0] + S[:, 1, 1] + 2 * off)


def W_interp(kgrid, Wgrid, kq):
    """Cubic interpolation of the (nk, 2, 2) bulk Hessian to arbitrary k."""
    kgrid, Wgrid = np.asarray(kgrid), np.asarray(Wgrid)
    nsp = Wgrid.shape[-1]
    out = np.zeros((len(kq), nsp, nsp))
    for a in range(nsp):
        for b in range(nsp):
            out[:, a, b] = CubicSpline(kgrid, Wgrid[:, a, b])(kq)
    return out


def bulk_structure(params, cfg, geom, k_md=None):
    """W(k) on the grid, S(k) on the grid and (optionally) at the MD shells,
    the thermodynamic factor, and the stability margin."""
    kg = np.asarray(geom.k)
    Wg = np.asarray(W_of_k(params, cfg, geom))
    out = dict(k=kg, W=Wg, S=S_from_W(Wg, kg, geom.nbar, geom.lB))
    out["Gamma"] = float(1.0 + 0.5 * geom.nbar * Wg[0].sum()) if Wg.shape[-1] == 2 else float(1.0 + geom.nbar * Wg[0, 0, 0])
    gf = geom if cfg.native_grid else fine_geometry(geom, 8)   # stability on a finer k grid
    Wf = jnp.asarray(W_of_k(params, cfg, gf))
    if not cfg.integrable:
        Wf = sym_W(Wf)
    lam = np.asarray(lambda_min_2x2(inverse_structure(Wf, gf)))
    kf = np.asarray(gf.k)
    out["lambda_min_rel"] = float((lam * geom.nbar).min())
    out["k_lambda_min"] = float(kf[np.argmin(lam)])
    out["n_unstable"] = int((lam < 0).sum())
    out["k_unstable"] = [float(kf[lam < 0].min()), float(kf[lam < 0].max())] if (lam < 0).any() else None
    if k_md is not None:
        Wq = W_interp(kg, Wg, k_md)
        out["S_md_k"] = S_from_W(Wq, np.asarray(k_md), geom.nbar, geom.lB)
    return out


def stillinger_lovett(params, cfg, geom, factor=32, mfit=6, deg=3):
    """Extrapolate S_ZZ(k)/(2 nbar) / (k^2/kappa_D^2) to k -> 0.  W(k) is a bulk
    property, so it is evaluated on a box `factor` times longer at the same
    nbar to reach small k; a polynomial in k^2 through the first `mfit` modes
    gives the intercept A, which must be 1 for any regular W."""
    if cfg.n_species != 2:
        return float("nan"), {}                     # no charge channel in a one-component fluid
    big = geom if cfg.native_grid else fine_geometry(geom, factor)
    ks = np.asarray(big.k)[1:mfit + 1]
    W = np.asarray(W_of_k(params, cfg, big, ks=ks))
    S = S_from_W(W, ks, geom.nbar, geom.lB)
    kD2 = 8.0 * np.pi * geom.lB * geom.nbar
    r = S["ZZ"] / (2 * geom.nbar) / (ks ** 2 / kD2)
    coef = np.polyfit(ks ** 2, r, deg)
    return float(coef[-1]), dict(k=ks, ratio=r)


# ------------------------------------------------------------ per-run
def heldout_report(params, cfg, batches_by_conc, el=True, mixing=0.1, log=print):
    """chi^2 per bin, per-run pulls, and (optionally) Euler-Lagrange profile
    predictions for every run in the batches.  Returns a list of dicts."""
    rows = []
    for c, (b, rs) in sorted(batches_by_conc.items()):
        pulls = np.asarray(residual_pulls(params, cfg, b))
        for i, r in enumerate(rs):
            labs = SPECIES[:pulls.shape[1]]
            row = dict(conc=c, tag=r.tag, family=r.family, kind=r.kind, status=r.status,
                       **{f"chi2_{lab}": float(np.mean(pulls[i, a] ** 2)) for a, lab in enumerate(labs)})
            row["chi2"] = 0.5 * (row["chi2_cation"] + row["chi2_anion"]) if len(labs) == 2 else row["chi2_cation"]
            if el:
                n_pred, info = el_solve(params, cfg, b.geom, r.V, mixing=mixing)
                row.update(el=info, profile=profile_metrics(n_pred, r.n), n_pred=n_pred)
            rows.append(row)
            if log:
                pm = row.get("profile")
                j = lambda key, fmt: "/".join(f"{pm[lab][key]:{fmt}}" for lab in labs)
                extra = (f"  L2 {j('rel_l2', '.3f')}  max {j('err_at_max', '+.3f')}  min {j('err_at_min', '+.3f')}"
                         f"  EL {'ok' if row['el']['converged'] else 'FAIL'} {row['el']['iters']}it"
                         if pm else "")
                log(f"    {r.key:12s} {r.family:7s} {r.kind:7s} chi2 "
                    + "/".join(f"{row[f'chi2_{lab}']:6.2f}" for lab in labs) + extra)
    return rows


def summarize(rows):
    if not rows:
        return {}
    labs = [lab for lab in SPECIES if f"chi2_{lab}" in rows[0]]
    out = dict(n_runs=len(rows), chi2=float(np.mean([r["chi2"] for r in rows])),
               **{f"chi2_{lab}": float(np.mean([r[f"chi2_{lab}"] for r in rows])) for lab in labs})
    if "profile" in rows[0]:
        for lab in labs:
            for key in ("rel_l2", "err_at_max", "err_at_min"):
                vals = [r["profile"][lab][key] for r in rows if r["el"]["converged"]]
                out[f"{lab}_{key}"] = float(np.mean(np.abs(vals))) if vals else float("nan")
        out["el_converged"] = int(sum(r["el"]["converged"] for r in rows))
    return out
