"""Structure-preserving neural density functional for the salt-in-polymer ions.

Implements doc/training_spec.md: learn F_ex[n_+, n_-] from planar MD profiles
by force matching, with the Coulomb part analytic and the ideal part exact.
Everything is JAX on a 1D periodic grid; float64 throughout.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
