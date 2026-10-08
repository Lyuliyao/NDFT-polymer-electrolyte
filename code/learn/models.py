from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations, product

import numpy as np
import jax
import jax.numpy as jnp

from .geometry import (E_SIGN, esign, gauss_conv, ddz, coulomb_phi, coulomb_dphi,
                       basis_widths, rfft, irfft)

ESIGN = jnp.asarray(E_SIGN)


def esign_j(nsp):

    return ESIGN if nsp == 2 else jnp.asarray(esign(nsp))


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "v3"
    M: int = 8
    s_min: float = 0.25
    s_max: float = 2.0
    depth: int = 1
    C: int = 4
    hidden: int = 64
    readout_hidden: tuple = (64, 64)
    grad_invariants: bool = False
    grad_channels: tuple = (2, 5)
    n_ref: float = 0.02
    deep_s_max: float = -1.0
    avni_a: float = 1.0
    pair_scale: float = 100.0
    packing: bool = False
    pack_width: float = 0.5
    pack_vmax: float = 4.0
    pack_eta_max: float = 0.95
    pack_v0: tuple = (-1.0, -1.0)
    log_inputs: bool = False

    win_bins: int = 30
    cace_a: int = 5
    cace_qcut: int = 3
    cace_scale: tuple = ()
    loc_hidden: tuple = (32, 16)
    kernel_net: tuple = ()
    activation: str = "silu"
    kernel_dr: float = 0.01
    kernel_rmax: float = 5.0

    cnn_channels: tuple = (16, 16, 32, 32, 64, 64)
    cnn_kernel: int = 3
    cnn_dilation: int = 2
    cnn_len: int = 3
    cnn_rows: int = 8
    seed: int = 0
    n_species: int = 2

    def widths(self):
        return basis_widths(self.M, self.s_min, self.s_max)

    def deep_widths(self):
        if self.deep_s_max > 0:
            return basis_widths(self.M, self.s_min, self.deep_s_max)
        return self.widths()

    @property
    def receptive_field(self):

        if self.variant == "c1win":
            return 0.1 * self.win_bins
        if self.variant == "cace":
            return 0.1 * self.cace_a * (self.cace_qcut + 0.5)
        return 3.0 * (self.s_max + (self.depth - 1) * (self.deep_s_max if self.deep_s_max > 0 else self.s_max))

    @property
    def native_grid(self):

        return self.variant == "cnn"

    @property
    def integrable(self):

        return self.variant != "c1win"

    @property
    def has_net(self):
        return self.variant in ("v2", "v2b", "v3")

    @property
    def has_pair(self):
        return self.variant in ("v1", "v2", "v2b")

    @property
    def has_pack(self):
        return self.packing or self.variant == "v2b"


def init_mlp(key, sizes, out_scale=1.0):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (a, b)) * jnp.sqrt(2.0 / (a + b))
        if i == len(sizes) - 2:
            W = W * out_scale
        ps.append({"W": W, "b": jnp.zeros(b)})
    return ps


ACTIVATIONS = {"silu": jax.nn.silu, "gelu": jax.nn.gelu, "tanh": jnp.tanh, "softplus": jax.nn.softplus}


def mlp(ps, x, act=jax.nn.silu):
    for i, p in enumerate(ps):
        x = x @ p["W"] + p["b"]
        if i < len(ps) - 1:
            x = act(x)
    return x


def lift(n, geom, widths):

    H = jax.vmap(lambda s: gauss_conv(n, s, geom.k))(jnp.asarray(widths))
    return jnp.transpose(H, (2, 1, 0)).reshape(n.shape[-1], -1)


def reconv(g, geom, widths):

    H = jax.vmap(lambda s: gauss_conv(g.T, s, geom.k))(jnp.asarray(widths))
    return jnp.transpose(H, (2, 1, 0)).reshape(g.shape[0], -1)


def grad_invariants(h0, geom, cfg):

    idx = [a * cfg.M + m for a in range(cfg.n_species) for m in cfg.grad_channels]
    dh = ddz(h0[:, idx].T, geom.k).T
    K = len(idx)
    iu = jnp.triu_indices(K)
    return (dh[:, :, None] * dh[:, None, :])[:, iu[0], iu[1]]


def net_dims(cfg):
    M2 = cfg.n_species * cfg.M
    gates = []
    for l in range(1, cfg.depth):
        d_in = M2 if l == 1 else cfg.M * cfg.C + M2
        gates.append((d_in, cfg.hidden, cfg.C))
    d_ro = M2 if cfg.depth == 1 else cfg.M * cfg.C + M2
    if cfg.grad_invariants:
        K = cfg.n_species * len(cfg.grad_channels)
        d_ro += K * (K + 1) // 2
    return gates, (d_ro, *cfg.readout_hidden, 1)


def init_net(cfg, key):
    gates, ro = net_dims(cfg)
    ps = {"gates": []}
    for sizes in gates:
        key, sub = jax.random.split(key)
        ps["gates"].append(init_mlp(sub, sizes))
    key, sub = jax.random.split(key)
    ps["readout"] = init_mlp(sub, ro, out_scale=1e-2)
    return ps


def basis_ft(knet, cfg, k):

    w = jnp.asarray(cfg.widths())
    dr = cfg.kernel_dr
    r = jnp.arange(0.5 * dr, cfg.kernel_rmax * cfg.s_max, dr)
    B = (2 * jnp.pi * w[None, :] ** 2) ** -1.5 * jnp.exp(-0.5 * (r[:, None] / w[None, :]) ** 2)
    K = B * (1.0 + mlp(knet, (r / cfg.s_max)[:, None], ACTIVATIONS[cfg.activation]))
    return 4 * jnp.pi * dr * jnp.einsum("rm,r,rk->mk", K, r * r, jnp.sinc(jnp.asarray(k)[None, :] * r[:, None] / jnp.pi))


def conv_basis(n, Kh):

    return irfft(rfft(n)[None] * Kh[:, None, :], n.shape[-1])


def net_F(ps, cfg, n, geom, Kh=None):
    if Kh is None:
        h0 = lift(n, geom, cfg.widths()) / cfg.n_ref
    else:
        h0 = jnp.transpose(conv_basis(n, Kh), (2, 1, 0)).reshape(n.shape[-1], -1) / cfg.n_ref
    if cfg.log_inputs:


        h0 = jnp.log(h0 + 1e-3)
    x = h0
    act = ACTIVATIONS[cfg.activation]
    for l, gate in enumerate(ps["gates"]):
        inp = h0 if l == 0 else jnp.concatenate([x, h0], axis=-1)
        x = reconv(mlp(gate, inp, act), geom, cfg.deep_widths())
    inp = h0 if cfg.depth == 1 else jnp.concatenate([x, h0], axis=-1)
    if cfg.grad_invariants:
        inp = jnp.concatenate([inp, grad_invariants(h0, geom, cfg)], axis=-1)
    phi = mlp(ps["readout"], inp, act)[:, 0]
    return geom.A * geom.dz * cfg.n_ref * jnp.sum(phi)


def pair_mu(c, cfg, n, geom, Kh=None):

    if Kh is None:
        conv = jax.vmap(lambda s: gauss_conv(n, s, geom.k))(jnp.asarray(cfg.widths()))
    else:
        conv = conv_basis(n, Kh)
    c = cfg.pair_scale * c
    if n.shape[0] == 1:
        return jnp.einsum("m,mn->n", c[0], conv[:, 0])[None]
    mu_p = jnp.einsum("m,mn->n", c[0], conv[:, 0]) + jnp.einsum("m,mn->n", c[2], conv[:, 1])
    mu_m = jnp.einsum("m,mn->n", c[1], conv[:, 1]) + jnp.einsum("m,mn->n", c[2], conv[:, 0])
    return jnp.stack([mu_p, mu_m])


def pair_F(c, cfg, n, geom, Kh=None):
    return 0.5 * geom.A * geom.dz * jnp.sum(n * pair_mu(c, cfg, n, geom, Kh))


def avni_uk(k, lB, a):

    k = jnp.asarray(k)
    safe = jnp.where(k > 0, k, 1.0)
    return jnp.where(k > 0, -4.0 * jnp.pi * lB * (1.0 - jnp.cos(safe * a)) / safe ** 2,
                     -2.0 * jnp.pi * lB * a * a)


def avni_mu(cfg, n, geom):
    uk = avni_uk(geom.k, geom.lB, cfg.avni_a)
    conv = irfft(uk * rfft(n[0] - n[1]), n.shape[-1])
    return ESIGN[:, None] * conv[None, :]


def avni_F(cfg, n, geom):
    return 0.5 * geom.A * geom.dz * jnp.sum(n * avni_mu(cfg, n, geom))


PACK_V0 = np.pi / 6.0 * np.array([0.4, 1.6]) ** 3


def pack_volumes(cfg, theta):
    return cfg.pack_vmax * jax.nn.sigmoid(theta)


def pack_F(theta, cfg, n, geom):

    h = gauss_conv(n, cfg.pack_width, geom.k)
    nw = h.sum(0)
    eta = jnp.einsum("a,an->n", pack_volumes(cfg, theta), h)
    eta = cfg.pack_eta_max * jnp.tanh(eta / cfg.pack_eta_max)
    return geom.A * geom.dz * jnp.sum(nw * (-jnp.log1p(-eta)))


def window_index(N, W):
    return (np.arange(N)[:, None] + np.arange(-W, W + 1)[None, :]) % N


def c1win_mu(ps, cfg, n):
    N = n.shape[-1]
    x = (n / cfg.n_ref)[:, window_index(N, cfg.win_bins)]
    return mlp(ps, jnp.transpose(x, (1, 0, 2)).reshape(N, -1)).T


MONOMIALS = [l for d in range(4) for l in np.ndindex(4, 4, 4) if sum(l) == d]


@lru_cache(maxsize=None)
def cace_tables(qcut, nsp=2):

    nm = len(MONOMIALS)
    idx = {l: i for i, l in enumerate(MONOMIALS)}
    ops = []
    for P in permutations(range(3)):
        Pinv = np.argsort(P)
        for s in product((1, -1), repeat=3):

            ops.append(([idx[tuple(l[Pinv[j]] for j in range(3))] for l in MONOMIALS],
                        [int(np.prod([s[i] ** l[i] for i in range(3)])) for l in MONOMIALS]))
    U = nsp * nm

    def unique(forms):
        keep = []
        for f in forms:
            f = np.asarray(f, float)
            if not np.any(f):
                continue
            if any(np.array_equal(f, g) or np.array_equal(f, -g) for g in keep):
                continue
            keep.append(f)
        return keep

    T1 = []
    for a in range(nsp):
        for l in range(nm):
            t = np.zeros(U)
            for perm, sign in ops:
                t[a * nm + perm[l]] += sign[l]
            T1.append(t)
    T2 = []
    for a, b in (((0, 0), (1, 1), (0, 1)) if nsp == 2 else [(a, b) for a in range(nsp) for b in range(a, nsp)]):
        for l1 in range(nm):
            for l2 in range(nm):
                T = np.zeros((U, U))
                for perm, sign in ops:
                    T[a * nm + perm[l1], b * nm + perm[l2]] += sign[l1] * sign[l2]
                T2.append(0.5 * (T + T.T))
    T1, T2 = np.array(unique(T1)), np.array(unique(T2))
    qz = np.arange(-qcut, qcut + 1)
    C = np.zeros((nm, len(qz)))
    r = np.arange(-qcut, qcut + 1)
    for j, z in enumerate(qz):
        for x in r:
            for y in r:
                if x * x + y * y + z * z > qcut * qcut or (x == 0 and y == 0 and z == 0):
                    continue
                for i, (lx, ly, lz) in enumerate(MONOMIALS):
                    C[i, j] += float(x) ** lx * float(y) ** ly * float(z) ** lz
    return qz, C, T1, T2


def voxel_average(n, a):

    return sum(jnp.roll(n, j, axis=-1) for j in range(-(a // 2), a // 2 + 1)) / a


def cace_features(cfg, n):

    rb = voxel_average(n, cfg.cace_a)
    x = rb / cfg.n_ref
    qz, C, T1, T2 = cace_tables(cfg.cace_qcut, cfg.n_species)
    sh = jnp.stack([jnp.roll(x, -int(q) * cfg.cace_a, axis=-1) for q in qz])
    A = jnp.einsum("qan,lq->aln", sh, jnp.asarray(C)).reshape(n.shape[0] * C.shape[0], -1)
    B1 = jnp.asarray(T1) @ A
    B2 = jnp.einsum("fuv,un,vn->fn", jnp.asarray(T2), A, A)
    return rb, jnp.concatenate([B1, B2]).T


def cace_nfeat(cfg):
    _, _, T1, T2 = cace_tables(cfg.cace_qcut, cfg.n_species)
    return len(T1) + len(T2)


def cace_feature_scale(cfg, batches):

    f = jax.jit(jax.vmap(lambda n: cace_features(cfg, n)[1]))
    ss, cnt = 0.0, 0
    for b in batches:
        B = np.asarray(f(b.n))
        ss, cnt = ss + (B ** 2).sum(axis=(0, 1)), cnt + B.shape[0] * B.shape[1]
    rms = np.sqrt(ss / cnt)
    return tuple(float(v) if v > 1e-12 * rms.max() else 1.0 for v in rms)


def cace_F(ps, cfg, n, geom):
    rb, B = cace_features(cfg, n)
    x = rb.T / cfg.n_ref
    scale = jnp.asarray(cfg.cace_scale) if cfg.cace_scale else 1.0
    a = mlp(ps["loc"], x) + mlp(ps["cace"], jnp.concatenate([x, B / scale], axis=-1))
    return geom.A * geom.dz * jnp.sum(rb.T * a)


def cnn_pooled_len(cfg, N):
    for _ in cfg.cnn_channels:
        N = N // 2
    return N


def cnn_applicable(cfg, geom):
    return cfg.variant != "cnn" or cnn_pooled_len(cfg, geom.z.shape[0]) == cfg.cnn_len


def init_cnn(cfg, key):
    ps, cin = {"conv": []}, cfg.n_species
    for cout in cfg.cnn_channels:
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (cout, cin, cfg.cnn_kernel)) * jnp.sqrt(2.0 / (cin * cfg.cnn_kernel + cout))
        ps["conv"].append({"W": W, "b": jnp.zeros(cout)})
        cin = cout
    nf = cin * cfg.cnn_len
    ps["head"] = {"W": jax.random.normal(key, (nf,)) * jnp.sqrt(1.0 / nf) * 1e-2, "b": jnp.zeros(())}
    return ps


def cnn_F(ps, cfg, n, geom):
    act = ACTIVATIONS[cfg.activation]
    x = (n / cfg.n_ref)[None]
    pad = cfg.cnn_dilation * (cfg.cnn_kernel - 1) // 2
    for layer in ps["conv"]:
        xp = jnp.concatenate([x[..., -pad:], x, x[..., :pad]], axis=-1)
        x = jax.lax.conv_general_dilated(xp, layer["W"], window_strides=(1,), padding="VALID",
                                         rhs_dilation=(cfg.cnn_dilation,),
                                         dimension_numbers=("NCH", "OIH", "NCH"))
        x = act(x + layer["b"][None, :, None])
        m = x.shape[-1] // 2
        x = 0.5 * (x[..., 0:2 * m:2] + x[..., 1:2 * m:2])
    if x.shape[-1] != cfg.cnn_len:
        raise ValueError(f"cnn: {geom.z.shape[0]} grid points pool to {x.shape[-1]} positions, "
                         f"the read-out was built for {cfg.cnn_len}")
    s = jnp.dot(x.reshape(-1), ps["head"]["W"]) + ps["head"]["b"]
    return geom.A * geom.L * cfg.n_ref * s


def init_params(cfg, key=None):
    key = jax.random.PRNGKey(cfg.seed) if key is None else key
    ps = {}
    if cfg.variant == "c1win":
        ps["c1"] = init_mlp(key, (cfg.n_species * (2 * cfg.win_bins + 1), *cfg.readout_hidden, cfg.n_species), out_scale=1e-2)
    if cfg.variant == "cnn":
        ps["cnn"] = init_cnn(cfg, key)
    if cfg.variant == "cace":
        k1, k2 = jax.random.split(key)
        ps["loc"] = init_mlp(k1, (cfg.n_species, *cfg.loc_hidden, cfg.n_species), out_scale=1e-2)
        ps["cace"] = init_mlp(k2, (cfg.n_species + cace_nfeat(cfg), *cfg.readout_hidden, cfg.n_species), out_scale=1e-2)
    if cfg.has_pair:
        ps["pair"] = jnp.zeros((3 if cfg.n_species == 2 else 1, cfg.M))
    if cfg.has_net:
        ps["net"] = init_net(cfg, key)
    if cfg.kernel_net:
        ps["knet"] = init_mlp(jax.random.fold_in(key, 7), (1, *cfg.kernel_net, cfg.M), out_scale=1e-2)
    if cfg.has_pack:
        if cfg.n_species != 2:
            raise NotImplementedError("the packing term is defined for the ion pair")
        v = np.where(np.array(cfg.pack_v0) > 0, np.array(cfg.pack_v0), PACK_V0)
        v0 = np.clip(v / cfg.pack_vmax, 1e-3, 1 - 1e-3)
        ps["pack"] = jnp.asarray(np.log(v0 / (1 - v0)))
    return ps


def F_ex(params, cfg, n, geom):
    if cfg.variant == "c1win":
        raise ValueError("c1win learns mu_theta directly and has no free energy")
    F = 0.0 * jnp.sum(n)
    if cfg.variant == "cace":
        F = F + cace_F(params, cfg, n, geom)
    if cfg.variant == "cnn":
        F = F + cnn_F(params["cnn"], cfg, n, geom)
    if cfg.variant == "avni":
        F = F + avni_F(cfg, n, geom)
    Kh = basis_ft(params["knet"], cfg, geom.k) if cfg.kernel_net else None
    if cfg.has_pair:
        F = F + pair_F(params["pair"], cfg, n, geom, Kh)
    if cfg.has_net:
        F = F + net_F(params["net"], cfg, n, geom, Kh)
    if cfg.has_pack:
        F = F + pack_F(params["pack"], cfg, n, geom)
    return F


def mu_theta(params, cfg, n, geom):

    if cfg.variant == "c1win":
        return c1win_mu(params["c1"], cfg, n)
    return jax.grad(F_ex, argnums=2)(params, cfg, n, geom) / (geom.A * geom.dz)


def mu_int(params, cfg, n, geom):
    return esign_j(n.shape[0])[:, None] * coulomb_phi(n, geom)[None, :] + mu_theta(params, cfg, n, geom)


def force_density(params, cfg, n, geom):

    dmu = esign_j(n.shape[0])[:, None] * coulomb_dphi(n, geom)[None, :] + ddz(mu_theta(params, cfg, n, geom), geom.k)
    return -n * dmu


def force_coulomb(n, geom):
    return -n * esign_j(n.shape[0])[:, None] * coulomb_dphi(n, geom)[None, :]


def W_of_k(params, cfg, geom, ks=None):

    N = geom.z.shape[0]; nsp = cfg.n_species
    n0 = jnp.full((nsp, N), geom.nbar)
    f = lambda n: mu_theta(params, cfg, n, geom)
    i0 = N // 2

    def one(beta, i0=i0):
        t = jnp.zeros((nsp, N)).at[beta, i0].set(1.0)
        _, dmu = jax.jvp(f, (n0,), (t,))
        return jnp.real(rfft(dmu) / rfft(t[beta])[None, :])

    if cfg.variant == "cnn":


        rows = jnp.arange(cfg.cnn_rows) * (N // cfg.cnn_rows)
        Wr = jax.vmap(lambda i: jax.vmap(lambda b: one(b, i))(jnp.arange(nsp)))(rows)
        W = jnp.transpose(jnp.mean(Wr, axis=0), (2, 1, 0))
    else:
        W = jnp.transpose(jax.vmap(one)(jnp.arange(nsp)), (2, 1, 0))
    if ks is not None:
        idx = jnp.rint(jnp.asarray(ks) * geom.L / (2 * jnp.pi)).astype(int)
        W = W[idx]
    return W


def lambda_min_2x2(Mx):
    if Mx.shape[-1] == 1:
        return Mx[..., 0, 0]
    a, b, d = Mx[..., 0, 0], Mx[..., 0, 1], Mx[..., 1, 1]


    return 0.5 * (a + d) - jnp.sqrt((0.5 * (a - d)) ** 2 + b * b + 1e-30)


def sym_W(W):

    return 0.5 * (W + jnp.swapaxes(W, -1, -2))


def inverse_structure(W, geom, ks=None):

    k = geom.k if ks is None else jnp.asarray(ks)
    k2 = jnp.where(k > 0, k * k, (2 * jnp.pi / geom.L) ** 2 * 1e-8)
    nsp = W.shape[-1]; e = esign_j(nsp)
    ee = jnp.outer(e, e)
    return W + jnp.eye(nsp)[None] / geom.nbar + (4 * jnp.pi * geom.lB / k2)[:, None, None] * ee[None]


def fine_geometry(geom, factor):

    from .geometry import make_geometry
    N = geom.z.shape[0]
    return make_geometry(geom.L * factor, N * factor, geom.n_pairs * factor ** 3, geom.lB, geom.conc)


def stability_penalty(params, cfg, geom, factor=4):

    g = geom if cfg.native_grid else fine_geometry(geom, factor)
    W = W_of_k(params, cfg, g)
    if not cfg.integrable:
        W = sym_W(W)
    return jnp.sum(jax.nn.relu(-lambda_min_2x2(inverse_structure(W, g)) * g.nbar))
