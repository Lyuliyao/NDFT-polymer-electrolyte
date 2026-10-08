from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

E_SIGN = np.array([1.0, -1.0])
SPECIES = ("cation", "anion")


def esign(nsp):

    return E_SIGN if nsp == 2 else np.zeros(nsp)


class Geometry(NamedTuple):

    L: float
    A: float
    dz: float
    z: jnp.ndarray
    k: jnp.ndarray
    nbar: float
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


def rfft(f):
    return jnp.fft.rfft(f, axis=-1)


def irfft(F, N):
    return jnp.fft.irfft(F, n=N, axis=-1)


def gauss_conv(f, s, k):

    N = f.shape[-1]
    return irfft(rfft(f) * jnp.exp(-0.5 * (k * s) ** 2), N)


def ddz(f, k):

    N = f.shape[-1]
    F = rfft(f) * (1j * k)
    if N % 2 == 0:
        F = F.at[..., -1].set(0.0)
    return irfft(F, N)


def _coulomb_k(n, geom):

    k = geom.k
    k2 = jnp.where(k > 0, k * k, 1.0)
    rho_z = n[0] - n[1] if n.shape[0] == 2 else jnp.zeros_like(n[0])
    F = 4.0 * jnp.pi * geom.lB * rfft(rho_z) / k2
    return F.at[0].set(0.0)


def coulomb_phi(n, geom):

    return irfft(_coulomb_k(n, geom), n.shape[-1])


def coulomb_dphi(n, geom):

    N = n.shape[-1]
    F = _coulomb_k(n, geom) * (1j * geom.k)
    if N % 2 == 0:
        F = F.at[-1].set(0.0)
    return irfft(F, N)


def coulomb_vk(k, lB):

    return 4.0 * np.pi * lB / np.where(k > 0, k * k, np.inf)


def basis_widths(M, s_min, s_max):
    return np.geomspace(s_min, s_max, M)


def basis_k(k, widths):

    return np.exp(-0.5 * (np.asarray(k)[None, :] * np.asarray(widths)[:, None]) ** 2)


def fold_periodic(y, q):

    y = np.asarray(y, float)
    F = np.fft.rfft(y, axis=-1)
    m = np.arange(F.shape[-1])
    F[..., m % q != 0] = 0.0
    return np.fft.irfft(F, n=y.shape[-1], axis=-1)
