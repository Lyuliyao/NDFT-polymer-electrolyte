import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .geometry import gauss_conv, ddz
from .models import force_density, force_coulomb, stability_penalty


def batch_loss(params, cfg, batch):

    pred = jax.vmap(lambda n: force_density(params, cfg, n, batch.geom))(batch.n)
    r = batch.f - pred
    loss = jnp.sum(batch.w * r * r)
    N = r.shape[-1]
    rk = jnp.fft.rfft(r, axis=-1)
    indices = jnp.arange(N // 2 + 1)
    endpoints = (indices == 0) | ((N % 2 == 0) & (indices == N // 2))
    mult = jnp.where(endpoints, 1.0, 2.0) / N
    power = mult * jnp.abs(rk) ** 2
    wbar = jnp.mean(batch.w, axis=-1)
    extra = batch.kboost[:, None, :] * power * wbar[:, :, None]
    return loss + jnp.sum(extra), r.size


def data_loss(params, cfg, batches):
    tot, cnt = 0.0, 0
    for b in batches:
        l, s = batch_loss(params, cfg, b)
        tot, cnt = tot + l, cnt + s
    return tot, cnt


def chi2_per_bin(params, cfg, batches):
    l, c = data_loss(params, cfg, batches)
    return l / max(c, 1)


def residual_pulls(params, cfg, batch):

    pred = jax.vmap(lambda n: force_density(params, cfg, n, batch.geom))(batch.n)
    return (batch.f - pred) * jnp.sqrt(batch.w)


def v1_design(cfg, batch):

    ws = jnp.asarray(cfg.widths())
    ys, Xs, wts = [], [], []
    for r in range(batch.n.shape[0]):
        n = batch.n[r]
        conv = jax.vmap(lambda s: gauss_conv(n, s, batch.geom.k))(ws)
        dconv = ddz(conv, batch.geom.k)
        y = np.asarray(batch.f[r] - force_coulomb(n, batch.geom))
        M, N = cfg.M, n.shape[-1]
        nsp = n.shape[0]; P = 3 if nsp == 2 else 1
        X = np.zeros((nsp, N, P, M))
        nn, dc = np.asarray(n), np.asarray(dconv)
        if nsp == 2:
            X[0, :, 0, :] = -nn[0][:, None] * dc[:, 0, :].T
            X[0, :, 2, :] = -nn[0][:, None] * dc[:, 1, :].T
            X[1, :, 1, :] = -nn[1][:, None] * dc[:, 1, :].T
            X[1, :, 2, :] = -nn[1][:, None] * dc[:, 0, :].T
        else:
            X[0, :, 0, :] = -nn[0][:, None] * dc[:, 0, :].T
        ys.append(y.reshape(-1)); Xs.append(X.reshape(nsp * N, P * M))
        wts.append(np.asarray(batch.w[r]).reshape(-1))

        b = np.asarray(batch.kboost[r])
        if np.any(b > 0):
            z = np.arange(N) * float(batch.geom.L) / N; ms = np.arange(1, N // 2 + 1); k = 2 * np.pi * ms / float(batch.geom.L)
            mult = np.where(ms == N // 2, 1.0 if N % 2 == 0 else 2.0, 2.0)
            lam = b[1:] * mult / N
            C, S_ = np.cos(np.outer(k, z)), np.sin(np.outer(k, z))
            for a in range(nsp):
                wbar = float(np.mean(np.asarray(batch.w[r, a])))
                G = np.vstack([np.sqrt(wbar * lam)[:, None] * C, np.sqrt(wbar * lam)[:, None] * S_])
                Xa = X[a].reshape(N, P * M); ya = y[a]
                ys.append(G @ ya); Xs.append(G @ Xa); wts.append(np.ones(G.shape[0]))
    return np.concatenate(ys), np.concatenate(Xs), np.concatenate(wts)


def fit_v1(cfg, batches, ridge=0.0):

    ys, Xs, ws = zip(*[v1_design(cfg, b) for b in batches])
    y, X, w = np.concatenate(ys), np.concatenate(Xs), np.concatenate(ws)
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    if ridge > 0:
        scale = np.sqrt(np.mean(Xw ** 2, axis=0))
        Xw = np.vstack([Xw, ridge * np.diag(scale)])
        yw = np.concatenate([yw, np.zeros(X.shape[1])])
    c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    sv = np.linalg.svd(X * sw[:, None], compute_uv=False)
    nb = sum(b.n.shape[1] * b.n.shape[2] * b.n.shape[0] for b in batches)
    chi2 = float(np.sum((yw[:nb] - Xw[:nb] @ c) ** 2) / nb)
    return {"pair": jnp.asarray(c.reshape(-1, cfg.M)) / cfg.pair_scale}, chi2, float(sv[0] / sv[-1])


def train(cfg, params, train_batches, heldout_batches, geoms, steps=4000, lr=1e-3,
          weight_decay=1e-5, stab_weight=100.0, eval_every=20, patience=60,
          log=print, min_steps=200, data_fn=None, monitor_fn=None, select="best"):

    train_batches = tuple(train_batches)
    heldout_batches = tuple(heldout_batches)
    geoms = tuple(geoms)
    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.01)
    opt = optax.adamw(sched, weight_decay=weight_decay)
    state = opt.init(params)

    def objective(p):
        l, cnt = data_loss(p, cfg, train_batches) if data_fn is None else data_fn(p)
        pen = 0.0
        if stab_weight > 0:
            for g in geoms:
                pen = pen + stability_penalty(p, cfg, g)
        return l + stab_weight * pen, (l / cnt, pen)

    @jax.jit
    def step(p, s):
        (obj, aux), grads = jax.value_and_grad(objective, has_aux=True)(p)
        upd, s = opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s, obj, aux

    if monitor_fn is not None:
        monitor = jax.jit(monitor_fn)
    elif heldout_batches:
        monitor = jax.jit(lambda p: chi2_per_bin(p, cfg, heldout_batches))
    else:
        monitor = jax.jit(lambda p: chi2_per_bin(p, cfg, train_batches))

    best, best_params, bad = np.inf, params, 0
    hist = []
    t0 = time.time()
    for it in range(steps):
        params, state, obj, (chi2_tr, pen) = step(params, state)
        if it % eval_every == 0 or it == steps - 1:
            ho = float(monitor(params))
            hist.append((it, float(chi2_tr), ho, float(pen)))
            if ho < best - 1e-12:
                best, best_params, bad = ho, params, 0
            else:
                bad += 1
            if it % (10 * eval_every) == 0:
                log(f"    step {it:5d}  train chi2/bin {float(chi2_tr):9.4f}  "
                    f"monitor {ho:9.4f}  stab {float(pen):.2e}  [{time.time()-t0:.0f}s]")
            if not np.isfinite(ho):
                log("    non-finite monitor; stopping")
                break
            if bad >= patience and it >= min_steps and select == "best":
                log(f"    early stop at step {it} (best monitor {best:.4f})")
                break
    if select == "last":
        return params, np.array(hist), float(monitor(params))
    return best_params, np.array(hist), best
