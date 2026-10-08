"""Model constants for the Tsamopoulos-Wang salt-in-polymer CG model.

All quantities in LJ units: energies in eps, lengths in sigma, time in tau.
Reference: ACS Macro Lett. 13, 322 (2024) / arXiv:2312.15401.
"""
import math

# ---------------------------------------------------------------- unit mapping
SIGMA_NM = 0.7          # sigma <-> 0.7 nm
T_REF_K = 400.0         # T* = 1.0 <-> 400 K
EPS_R = 7.5             # relative permittivity (uniform)

# physical constants (SI, CODATA 2018)
_E = 1.602176634e-19
_EPS0 = 8.8541878128e-12
_KB = 1.380649e-23


def bjerrum_sigma(Tstar=1.0, eps_r=EPS_R):
    """Bjerrum length in units of sigma at reduced temperature Tstar."""
    lb_vac_m = _E ** 2 / (4.0 * math.pi * _EPS0 * _KB * (T_REF_K * Tstar))
    return lb_vac_m / (eps_r * SIGMA_NM * 1e-9)


# Reduced charge: defined so that l_B = q*^2 / (eps_r T*) reproduces the
# physical Bjerrum length.  With LAMMPS `dielectric 7.5` and q = +-QSTAR the
# pair energy is q_i q_j / (eps_r r) in units of eps.
QSTAR = math.sqrt(bjerrum_sigma(1.0) * EPS_R * 1.0)

# ------------------------------------------------------------------- topology
N_CHAINS = 400
N_MONO = 30
N_BEAD = N_CHAINS * N_MONO        # 12000 polymer beads

# atom types
T_MONO, T_CAT, T_ANI = 1, 2, 3
SIGMA_T = {T_MONO: 1.0, T_CAT: 0.4, T_ANI: 1.6}
CHARGE_T = {T_MONO: 0.0, T_CAT: +QSTAR, T_ANI: -QSTAR}
MASS_T = {T_MONO: 1.0, T_CAT: 1.0, T_ANI: 1.0}

# ------------------------------------------------------------------- energetics
EPSILON = 1.0                     # eps_ij = eps for all pairs
FENE_K, FENE_R0 = 30.0, 1.5
CUT_MM = 2.0                      # monomer-monomer LJ cutoff (attractive tail kept)
WCA = 2.0 ** (1.0 / 6.0)
S_SOLV = 4.33                     # solvation prefactor, ion-monomer only
RC_SOLV = 5.0
# Length in the solvation term U = -S[(s/r)^4 - (s/rc)^4]: "one" puts s = 1 for
# every ion-monomer pair (Hall's bornsolv convention), "ij" puts s = sigma_ij as
# eq. (3) is printed.  Only "one" reproduces the SI's V, D+, D- and sigma
# (README, "Solvation screen"), so it is the production model.
SOLV_SIGMA = "one"

DT = 0.005
TDAMP = 1.0


def sigma_ij(i, j):
    return 0.5 * (SIGMA_T[i] + SIGMA_T[j])


def lj_cut(i, j):
    """LJ cutoff for pair (i,j): full 2.0 sigma only for monomer-monomer."""
    if i == T_MONO and j == T_MONO:
        return CUT_MM
    return WCA * sigma_ij(i, j)


def is_solvated(i, j):
    """Solvation 1/r^4 acts on ion-monomer pairs only."""
    return (i == T_MONO) != (j == T_MONO)


def solv_sigma(i, j, mode=None):
    """Length s in the solvation term of pair (i,j), for SOLV_SIGMA or `mode`."""
    return 1.0 if (mode or SOLV_SIGMA) == "one" else sigma_ij(i, j)


def n_pairs(conc):
    """Number of ion pairs for c_LJ = N_+/N_bead."""
    n = conc * N_BEAD
    assert abs(n - round(n)) < 1e-9, f"c={conc} does not give an integer ion count"
    return int(round(n))


if __name__ == "__main__":
    print(f"q*            = {QSTAR:.4f}")
    for T in (1.0, 0.8, 0.6):
        print(f"l_B(T*={T})   = {bjerrum_sigma(T):.3f} sigma")
    print(f"contact +/- E = {-QSTAR**2/(EPS_R*sigma_ij(T_CAT,T_ANI)):.3f} eps at r=1 sigma")
    for (i, j) in [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]:
        print(f"  pair {i}{j}: sigma_ij={sigma_ij(i,j):.4f}  rc_lj={lj_cut(i,j):.6f}"
              f"  solv={is_solvated(i,j)}")
    for c in (0.02, 0.04, 0.06, 0.08):
        print(f"  c={c}: {n_pairs(c)} ion pairs")
