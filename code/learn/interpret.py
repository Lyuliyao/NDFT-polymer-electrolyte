import numpy as np


def W_r_from_k(k, Wk, r):

    kr = np.outer(r, k)
    j0 = np.where(kr > 1e-12, np.sin(kr) / np.where(kr > 1e-12, kr, 1.0), 1.0)
    return np.trapezoid(k ** 2 * Wk[None, :] * j0, k, axis=1) / (2 * np.pi ** 2)
