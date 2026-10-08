"""The functional variants (spec §3) and the baselines (spec §6).

F_ex[n] is a scalar of the density arrays n (2, N); mu_theta = grad_n F /(A dz)
and the force density -n_a d_z(e_a phi + mu_theta_a) follow by autodiff.
All variants are built from radial kernels (Gaussians, or the analytic
cut-off Coulomb kernel) and pointwise maps, so they are translation
equivariant scalar functionals on any periodic grid.
"""
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
    """Charges of nsp species as a jnp array: ESIGN for the ion pair (the unchanged path), zeros for one species."""
    return ESIGN if nsp == 2 else jnp.asarray(esign(nsp))


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "v3"            # pb | avni | v1 | v2 | v3
    M: int = 8
    s_min: float = 0.25
    s_max: float = 2.0
    depth: int = 1                 # L; L=1 is the readout on h^(0) alone
    C: int = 4                     # gated channels per re-convolution layer
    hidden: int = 64
    readout_hidden: tuple = (64, 64)
    grad_invariants: bool = False
    grad_channels: tuple = (2, 5)  # basis indices whose d_z h products are fed to Phi
    n_ref: float = 0.02            # fixed input/output scale of the network
    deep_s_max: float = -1.0       # widths of layers l>=1; <0 means the same basis
    avni_a: float = 1.0
    pair_scale: float = 100.0      # c_abm = pair_scale * params["pair"]; O(100) kT sigma^3 is natural
    packing: bool = False          # add int n_w [-ln(1 - eta)], eta = sum_a v_a (B * n_a); "v2b" sets it
    pack_width: float = 0.5        # width of the Gaussian that defines the local densities of the packing term
    pack_vmax: float = 4.0         # v_a = pack_vmax * sigmoid(theta_a)
    pack_eta_max: float = 0.95     # eta is soft-clipped below this
    pack_v0: tuple = (-1.0, -1.0)  # initial v_a; negative = bare ion volumes pi/6 sigma^3
    log_inputs: bool = False       # feed log(h/n_ref + 1e-3) to the network instead of h/n_ref
    # architectures of earlier neural functionals, for the comparison of structure
    win_bins: int = 30             # c1win: half-width of the density window, in bins (~0.1 sigma each)
    cace_a: int = 5                # cace: lattice spacing (voxel size) in bins, odd
    cace_qcut: int = 3             # cace: stencil radius in lattice spacings
    cace_scale: tuple = ()         # cace: per-invariant scales, the rms over the training runs
    loc_hidden: tuple = (32, 16)   # cace: hidden widths of the pointwise baseline a_loc
    kernel_net: tuple = ()         # hidden widths of g(r): basis kernels B_m(r) [1 + g_m(r)]; () = plain Gaussians
    activation: str = "silu"       # of the readout Phi and of g: silu | gelu | tanh | softplus
    kernel_dr: float = 0.01        # radial step of the quadrature for their Fourier transforms
    kernel_rmax: float = 5.0       # quadrature range in units of s_max
    # cnn: the global convolutional free energy of Dijkman et al. (PRL 134, 056103, 2025)
    cnn_channels: tuple = (16, 16, 32, 32, 64, 64)
    cnn_kernel: int = 3
    cnn_dilation: int = 2
    cnn_len: int = 3               # positions left after the poolings, fixed by the box grid
    cnn_rows: int = 8              # Hessian rows averaged for W(k) (the model is not translation invariant)
    seed: int = 0
    n_species: int = 2             # 2: the ion pair (+, -); 1: a one-component fluid (no charges, one kernel per basis)

    def widths(self):
        return basis_widths(self.M, self.s_min, self.s_max)

    def deep_widths(self):
        if self.deep_s_max > 0:
            return basis_widths(self.M, self.s_min, self.deep_s_max)
        return self.widths()

    @property
    def receptive_field(self):
        """~3 s_max per convolution layer (baselines: the window or stencil radius at 0.1 sigma per bin)."""
        if self.variant == "c1win":
            return 0.1 * self.win_bins
        if self.variant == "cace":
            return 0.1 * self.cace_a * (self.cace_qcut + 0.5)
        return 3.0 * (self.s_max + (self.depth - 1) * (self.deep_s_max if self.deep_s_max > 0 else self.s_max))

    @property
    def native_grid(self):
        """True for the CNN, whose read-out fixes the number of grid points: it cannot be
        evaluated on a longer box or a finer k grid, only on the box it was trained on."""
        return self.variant == "cnn"

    @property
    def integrable(self):
        """False for the one-body network, which has no free energy."""
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


# ----------------------------------------------------------------- pointwise MLP
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


# ------------------------------------------------------------------ layers
def lift(n, geom, widths):
    """h^(0): (N, 2M), channel index alpha*M + m = B_m * n_alpha."""
    H = jax.vmap(lambda s: gauss_conv(n, s, geom.k))(jnp.asarray(widths))   # (M, 2, N)
    return jnp.transpose(H, (2, 1, 0)).reshape(n.shape[-1], -1)


def reconv(g, geom, widths):
    """(N, C) -> (N, C*M): every channel convolved with every basis kernel."""
    H = jax.vmap(lambda s: gauss_conv(g.T, s, geom.k))(jnp.asarray(widths))  # (M, C, N)
    return jnp.transpose(H, (2, 1, 0)).reshape(g.shape[0], -1)


def grad_invariants(h0, geom, cfg):
    """Parity-even products d_z h_i d_z h_j for the selected channels."""
    idx = [a * cfg.M + m for a in range(cfg.n_species) for m in cfg.grad_channels]
    dh = ddz(h0[:, idx].T, geom.k).T                                       # (N, K)
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
    """(M, nk) three-dimensional Fourier transforms of the learned radial kernels
    K_m(r) = B_m(r) [1 + g_m(r)], B_m the normalized Gaussians and g a perceptron of
    r / s_max: K_m(k) = 4 pi int r^2 K_m(r) sin(kr)/(kr) dr.  Radial, so every
    structural property of the functional is kept; g = 0 gives the Gaussian basis."""
    w = jnp.asarray(cfg.widths())
    dr = cfg.kernel_dr
    r = jnp.arange(0.5 * dr, cfg.kernel_rmax * cfg.s_max, dr)
    B = (2 * jnp.pi * w[None, :] ** 2) ** -1.5 * jnp.exp(-0.5 * (r[:, None] / w[None, :]) ** 2)    # (nr, M)
    K = B * (1.0 + mlp(knet, (r / cfg.s_max)[:, None], ACTIVATIONS[cfg.activation]))
    return 4 * jnp.pi * dr * jnp.einsum("rm,r,rk->mk", K, r * r, jnp.sinc(jnp.asarray(k)[None, :] * r[:, None] / jnp.pi))


def conv_basis(n, Kh):
    """(M, 2, N): every species convolved with every basis kernel, given its transform."""
    return irfft(rfft(n)[None] * Kh[:, None, :], n.shape[-1])


def net_F(ps, cfg, n, geom, Kh=None):
    if Kh is None:
        h0 = lift(n, geom, cfg.widths()) / cfg.n_ref
    else:
        h0 = jnp.transpose(conv_basis(n, Kh), (2, 1, 0)).reshape(n.shape[-1], -1) / cfg.n_ref
    if cfg.log_inputs:
        # the lifted densities are positive (Gaussian smoothing of n >= 0), so
        # the log is defined; mixed channels of deeper layers are not logged
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


# -------------------------------------------------------------- pair kernels
def pair_mu(c, cfg, n, geom, Kh=None):
    """mu_a = sum_b u_ab * n_b for u_ab = sum_m c_abm B_m; c is (3, M) in the
    order (++, --, +-), or (1, M) for the single kernel of a one-component fluid."""
    if Kh is None:
        conv = jax.vmap(lambda s: gauss_conv(n, s, geom.k))(jnp.asarray(cfg.widths()))  # (M, 2, N)
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
    """Planar/3D Fourier transform of u = -theta(a-r) l_B/r (the e_a e_b sign
    is applied by the caller): -4 pi l_B (1 - cos ka)/k^2, -> -2 pi l_B a^2."""
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


# ------------------------------------------------------------- packing term
PACK_V0 = np.pi / 6.0 * np.array([0.4, 1.6]) ** 3      # ion volumes, the initial v_a


def pack_volumes(cfg, theta):
    return cfg.pack_vmax * jax.nn.sigmoid(theta)


def pack_F(theta, cfg, n, geom):
    """int d^3r n_w(r) [-ln(1 - eta(r))], n_w = sum_a (B * n_a), eta = sum_a v_a (B * n_a),
    with B a Gaussian of width pack_width and v_a > 0 learnable.  A fixed
    functional form with the excluded-volume asymptotics (it diverges as eta
    -> 1), so that the growth of Gamma with concentration is carried by
    physics rather than by the tail of the network's activation.  Radial
    kernel plus pointwise scalar, so all structural properties are kept."""
    h = gauss_conv(n, cfg.pack_width, geom.k)                # (2, N)
    nw = h.sum(0)
    eta = jnp.einsum("a,an->n", pack_volumes(cfg, theta), h)
    eta = cfg.pack_eta_max * jnp.tanh(eta / cfg.pack_eta_max)
    return geom.A * geom.dz * jnp.sum(nw * (-jnp.log1p(-eta)))


# ------------------------------------ baseline: one-body direct correlation
# Local learning of c1 (Sammuller et al., PNAS 120, e2312484120, 2023), with
# the electrostatics kept analytic as for ions in Bui & Cox (2025): a
# perceptron maps the densities of both species in a window of +-win_bins
# bins around z to mu_theta(z) = -c1(z) directly.  No free energy is formed,
# so integrability, the Noether sum rules and reflection symmetry hold only
# as well as the fit.
def window_index(N, W):
    return (np.arange(N)[:, None] + np.arange(-W, W + 1)[None, :]) % N


def c1win_mu(ps, cfg, n):
    N = n.shape[-1]
    x = (n / cfg.n_ref)[:, window_index(N, cfg.win_bins)]                    # (2, N, 2W+1)
    return mlp(ps, jnp.transpose(x, (1, 0, 2)).reshape(N, -1)).T            # (2, N)


# ------------------------------- baseline: equivariant local free energy
# Equi-cDFT (Cheng, arXiv:2608.13506, Eqs. 2-5 and 15): F = dV sum_g rho_g
# [a_loc(rho_g) + a_CACE(rho_g, B_g)] on a cubic lattice of voxels, where B_g
# are the O_h-invariant products (order nu <= 2) of the Cartesian moments
# A_l(g) = sum_q rho_{g+q} q_x^lx q_y^ly q_z^lz, |l| <= 3, over the integer
# offsets 0 < |q| <= qcut (one constant radial channel).  Two species:
# moments per species, products within and across species, a_alpha per
# species.  A planar profile on a lattice of spacing a = cace_a bins: the
# voxel density is the box average over a, the lateral sums of the stencil
# are done exactly, and the functional is averaged over the a registrations
# of the lattice on the bin grid (it is evaluated at every bin).
MONOMIALS = [l for d in range(4) for l in np.ndindex(4, 4, 4) if sum(l) == d]   # the 20 l with |l| <= 3


@lru_cache(maxsize=None)
def cace_tables(qcut, nsp=2):
    """Stencil offsets q_z, lateral sums C (20, nq), and the invariant forms:
    T1 (F1, 40) for B = T1 A and T2 (F2, 40, 40) for B = A.T2.A, where A holds
    the moments of both species (index alpha*20 + l).  Group sums over the 48
    signed permutations; vanishing forms and duplicates (up to sign) removed."""
    nm = len(MONOMIALS)
    idx = {l: i for i, l in enumerate(MONOMIALS)}
    ops = []
    for P in permutations(range(3)):
        Pinv = np.argsort(P)
        for s in product((1, -1), repeat=3):
            # (R q)_i = s_i q_P(i): (R q)^l = prod_i s_i^l_i * prod_j q_j^l_Pinv(j)
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
    """Box average over a (odd) bins centred on each bin."""
    return sum(jnp.roll(n, j, axis=-1) for j in range(-(a // 2), a // 2 + 1)) / a


def cace_features(cfg, n):
    """Voxel densities (2, N) and the raw invariants (N, F1 + F2)."""
    rb = voxel_average(n, cfg.cace_a)
    x = rb / cfg.n_ref
    qz, C, T1, T2 = cace_tables(cfg.cace_qcut, cfg.n_species)
    sh = jnp.stack([jnp.roll(x, -int(q) * cfg.cace_a, axis=-1) for q in qz])  # x(z + q a), (nq, nsp, N)
    A = jnp.einsum("qan,lq->aln", sh, jnp.asarray(C)).reshape(n.shape[0] * C.shape[0], -1)
    B1 = jnp.asarray(T1) @ A
    B2 = jnp.einsum("fuv,un,vn->fn", jnp.asarray(T2), A, A)
    return rb, jnp.concatenate([B1, B2]).T


def cace_nfeat(cfg):
    _, _, T1, T2 = cace_tables(cfg.cace_qcut, cfg.n_species)
    return len(T1) + len(T2)


def cace_feature_scale(cfg, batches):
    """rms of each invariant over the MD profiles of the training runs (1 where
    an invariant vanishes identically for planar profiles)."""
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
    a = mlp(ps["loc"], x) + mlp(ps["cace"], jnp.concatenate([x, B / scale], axis=-1))   # (N, 2)
    return geom.A * geom.dz * jnp.sum(rb.T * a)


# ------------------------------------------ baseline: global CNN free energy
# Dijkman et al., PRL 134, 056103 (2025), End Matter and Supplemental Material: the excess
# free energy is a scalar CNN of the planar density profile, six periodic dilated 1D
# convolutions (kernel 3, dilation 2; the main text says 3), channels 16, 16, 32, 32, 64, 64,
# average pooling with kernel 2 after each layer, one scalar output.  Their 24.4K parameters
# for n = 320 grid points fix the read-out: a linear layer on the flattened last feature map
# (64 x 320/64 = 320 inputs; a global sum would give 24.1K).  The activation is not stated;
# softplus here (smooth, as the Hessian needs; the same as our functional).  Here two input
# channels (n_+, n_-) and the analytic Coulomb term are added.  F is a scalar, so the model
# is integrable, but the pooling strides and the read-out break translation and reflection
# invariance, and the read-out ties the model to its grid: floor-pooling six times leaves
# cnn_len = 3 positions for every grid of 192-255 points (all training boxes here).
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
    x = (n / cfg.n_ref)[None]                                            # (1, 2, N): batch, channel, z
    pad = cfg.cnn_dilation * (cfg.cnn_kernel - 1) // 2
    for layer in ps["conv"]:
        xp = jnp.concatenate([x[..., -pad:], x, x[..., :pad]], axis=-1)  # periodic padding
        x = jax.lax.conv_general_dilated(xp, layer["W"], window_strides=(1,), padding="VALID",
                                         rhs_dilation=(cfg.cnn_dilation,),
                                         dimension_numbers=("NCH", "OIH", "NCH"))
        x = act(x + layer["b"][None, :, None])
        m = x.shape[-1] // 2
        x = 0.5 * (x[..., 0:2 * m:2] + x[..., 1:2 * m:2])                # average pooling, kernel 2, floor
    if x.shape[-1] != cfg.cnn_len:
        raise ValueError(f"cnn: {geom.z.shape[0]} grid points pool to {x.shape[-1]} positions, "
                         f"the read-out was built for {cfg.cnn_len}")
    s = jnp.dot(x.reshape(-1), ps["head"]["W"]) + ps["head"]["b"]
    return geom.A * geom.L * cfg.n_ref * s


# ------------------------------------------------------------------- F, mu, f
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
    """delta F_ex / delta n_a (z), (2, N); the network output itself for c1win."""
    if cfg.variant == "c1win":
        return c1win_mu(params["c1"], cfg, n)
    return jax.grad(F_ex, argnums=2)(params, cfg, n, geom) / (geom.A * geom.dz)


def mu_int(params, cfg, n, geom):
    return esign_j(n.shape[0])[:, None] * coulomb_phi(n, geom)[None, :] + mu_theta(params, cfg, n, geom)


def force_density(params, cfg, n, geom):
    """-n_a d_z (e_a phi[n] + mu_theta_a[n]), the model side of the training identity."""
    dmu = esign_j(n.shape[0])[:, None] * coulomb_dphi(n, geom)[None, :] + ddz(mu_theta(params, cfg, n, geom), geom.k)
    return -n * dmu


def force_coulomb(n, geom):
    return -n * esign_j(n.shape[0])[:, None] * coulomb_dphi(n, geom)[None, :]


# ------------------------------------------------------------ bulk Hessian
def W_of_k(params, cfg, geom, ks=None):
    """Bulk Hessian W_ab(k) of F_ex at the uniform density nbar.

    The response of mu_theta to a perturbation of the uniform state is linear
    in the perturbation (that is what a Jacobian-vector product computes), and
    by translation invariance the response to cos(kz) is W(k) cos(kz).  So a
    single delta-function tangent per species gives the real-space kernel
    W_ab(z - z0), and its rfft divided by that of the delta gives W_ab(k) at
    every grid k at once: two JVPs per state point instead of one per k.
    Returns (nk, 2, 2) on the rfft grid, or at the given grid k's (which must
    be multiples of 2 pi / L)."""
    N = geom.z.shape[0]; nsp = cfg.n_species
    n0 = jnp.full((nsp, N), geom.nbar)
    f = lambda n: mu_theta(params, cfg, n, geom)
    i0 = N // 2

    def one(beta, i0=i0):
        t = jnp.zeros((nsp, N)).at[beta, i0].set(1.0)
        _, dmu = jax.jvp(f, (n0,), (t,))
        return jnp.real(rfft(dmu) / rfft(t[beta])[None, :])        # (2, nk) = W[:, beta](k)

    if cfg.variant == "cnn":
        # not translation invariant: the Fourier diagonal of the Hessian, estimated as the
        # mean over cnn_rows phase-aligned rows (the division by rfft of the delta aligns them)
        rows = jnp.arange(cfg.cnn_rows) * (N // cfg.cnn_rows)
        Wr = jax.vmap(lambda i: jax.vmap(lambda b: one(b, i))(jnp.arange(nsp)))(rows)   # (rows, beta, alpha, nk)
        W = jnp.transpose(jnp.mean(Wr, axis=0), (2, 1, 0))
    else:
        W = jnp.transpose(jax.vmap(one)(jnp.arange(nsp)), (2, 1, 0))     # (nk, alpha, beta)
    if ks is not None:
        idx = jnp.rint(jnp.asarray(ks) * geom.L / (2 * jnp.pi)).astype(int)
        W = W[idx]
    return W


def lambda_min_2x2(Mx):
    if Mx.shape[-1] == 1:                     # one species: the matrix is the number itself
        return Mx[..., 0, 0]
    a, b, d = Mx[..., 0, 0], Mx[..., 0, 1], Mx[..., 1, 1]
    # + 1e-30: without charges (LJ, l_B = 0) the two identical labels can make the
    # matrix exactly degenerate, where the derivative of the square root is 0 * inf
    return 0.5 * (a + d) - jnp.sqrt((0.5 * (a - d)) ** 2 + b * b + 1e-30)


def sym_W(W):
    """Symmetric part in the species indices: the stability test of a model
    without a free energy (c1win) uses it, since W_+- != W_-+ there."""
    return 0.5 * (W + jnp.swapaxes(W, -1, -2))


def inverse_structure(W, geom, ks=None):
    """S^-1(k) = nbar^-1 delta_ab + 4 pi l_B e_a e_b / k^2 + W_ab(k), (nk, 2, 2).
    The Coulomb term belongs in the stability test: the exact functional's W
    cancels part of the mean field at small k (the learned V1 kernel has
    W_++(0) = -165, W_+-(0) = +125 at c = 0.04 while its S(k) matches MD),
    so nbar^-1 + W alone is indefinite there although S(k) is positive.  At
    k = 0 the charge channel is infinitely stiff, so only the number channel
    is tested."""
    k = geom.k if ks is None else jnp.asarray(ks)
    k2 = jnp.where(k > 0, k * k, (2 * jnp.pi / geom.L) ** 2 * 1e-8)
    nsp = W.shape[-1]; e = esign_j(nsp)
    ee = jnp.outer(e, e)
    return W + jnp.eye(nsp)[None] / geom.nbar + (4 * jnp.pi * geom.lB / k2)[:, None, None] * ee[None]


def fine_geometry(geom, factor):
    """The same state point in a box `factor` times longer: W(k) is a bulk
    property, so this only refines the k grid (spacing 2 pi / (factor L))."""
    from .geometry import make_geometry
    N = geom.z.shape[0]
    return make_geometry(geom.L * factor, N * factor, geom.n_pairs * factor ** 3, geom.lB, geom.conc)


def stability_penalty(params, cfg, geom, factor=4):
    """sum_k relu(-lambda_min[S^-1(k)]) * nbar (dimensionless), on a k grid
    `factor` times finer than the box's: an instability between two box
    wavevectors (a pole of S(k) at, say, k = 0.41 with spacing 0.26) would
    otherwise go unseen."""
    g = geom if cfg.native_grid else fine_geometry(geom, factor)
    W = W_of_k(params, cfg, g)
    if not cfg.integrable:
        W = sym_W(W)
    return jnp.sum(jax.nn.relu(-lambda_min_2x2(inverse_structure(W, g)) * g.nbar))
