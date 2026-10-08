"""Grid, Fourier helpers, the analytic Coulomb part, Gaussian basis, folding.

Species order everywhere: index 0 = cation (+), index 1 = anion (-).
Units: k_B T = 1, lengths in sigma, e_± = ±1, l_B = q*^2/eps_r.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

E_SIGN = np.array([1.0, -1.0])
SPECIES = ("cation", "anion")       # species order of the data files; a one-component fluid stores its single species as "cation"


def esign(nsp):
    """Charges e_a of nsp species: (+1, -1) for the ion pair, 0 for the single species of a one-component fluid."""
    return E_SIGN if nsp == 2 else np.zeros(nsp)


class Geometry(NamedTuple):
    """One state point's box and grid.  A pytree, so it can be passed to jit."""
    L: float
    A: float
    dz: float
    z: jnp.ndarray       # bin centres (N,)
    k: jnp.ndarray       # rfft wavenumbers (N//2+1,)
    nbar: float          # N_pm / L^3
    n_pairs: float
    lB: float
    conc: float

    @property
    def N(self):
        return self.z.shape[0]


def make_geometry(L, N, n_pairs, lB, conc):
    dz = L / N
    z = (np.arange(N) + 0.5) * dz
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dz)
    return Geometry(L=float(L), A=float(L) ** 2, dz=float(dz), z=jnp.asarray(z),
                    k=jnp.asarray(k), nbar=float(n_pairs) / float(L) ** 3,
                    n_pairs=float(n_pairs), lB=float(lB), conc=float(conc))


# ----------------------------------------------------------------- Fourier
def rfft(f):
    return jnp.fft.rfft(f, axis=-1)


def irfft(F, N):
    return jnp.fft.irfft(F, n=N, axis=-1)


def gauss_conv(f, s, k):
    """Periodic convolution of f (..., N) with the normalized 1D Gaussian of
    width s: exact in k-space, exp(-k^2 s^2/2), which is also the planar
    reduction of the normalized 3D Gaussian B_m."""
    N = f.shape[-1]
    return irfft(rfft(f) * jnp.exp(-0.5 * (k * s) ** 2), N)


def ddz(f, k):
    """Spectral derivative of a periodic field (..., N)."""
    N = f.shape[-1]
    F = rfft(f) * (1j * k)
    if N % 2 == 0:
        F = F.at[..., -1].set(0.0)
    return irfft(F, N)


# ----------------------------------------------------------------- Coulomb
def _coulomb_k(n, geom):
    """4 pi l_B rho_Z(k) / k^2, with the k = 0 mode dropped (electroneutral)."""
    k = geom.k
    k2 = jnp.where(k > 0, k * k, 1.0)
    rho_z = n[0] - n[1] if n.shape[0] == 2 else jnp.zeros_like(n[0])      # no charges in a one-component fluid
    F = 4.0 * jnp.pi * geom.lB * rfft(rho_z) / k2
    return F.at[0].set(0.0)


def coulomb_phi(n, geom):
    """Mean-field electrostatic potential phi[n](z), k_B T / e units."""
    return irfft(_coulomb_k(n, geom), n.shape[-1])


def coulomb_dphi(n, geom):
    """d phi / dz."""
    N = n.shape[-1]
    F = _coulomb_k(n, geom) * (1j * geom.k)
    if N % 2 == 0:
        F = F.at[-1].set(0.0)
    return irfft(F, N)


def coulomb_vk(k, lB):
    """v_Coul(k) = 4 pi l_B / k^2 (the e_a e_b factor applied by the caller)."""
    return 4.0 * np.pi * lB / np.where(k > 0, k * k, np.inf)


# ------------------------------------------------------------------- basis
def basis_widths(M, s_min, s_max):
    return np.geomspace(s_min, s_max, M)


def basis_k(k, widths):
    """Fourier transforms of the normalized Gaussians, (M, nk)."""
    return np.exp(-0.5 * (np.asarray(k)[None, :] * np.asarray(widths)[:, None]) ** 2)


# ----------------------------------------------------------------- folding
def fold_periodic(y, q):
    """Project y(z) (..., N) onto functions of period L/q: keep only the
    harmonics m with m % q == 0.  This is the average over the q translates,
    done exactly even when q does not divide the number of bins."""
    y = np.asarray(y, float)
    F = np.fft.rfft(y, axis=-1)
    m = np.arange(F.shape[-1])
    F[..., m % q != 0] = 0.0
    return np.fft.irfft(F, n=y.shape[-1], axis=-1)
