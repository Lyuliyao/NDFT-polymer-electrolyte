import math


SIGMA_NM = 0.7
T_REF_K = 400.0
EPS_R = 7.5


_E = 1.602176634e-19
_EPS0 = 8.8541878128e-12
_KB = 1.380649e-23


def bjerrum_sigma(Tstar=1.0, eps_r=EPS_R):

    lb_vac_m = _E ** 2 / (4.0 * math.pi * _EPS0 * _KB * (T_REF_K * Tstar))
    return lb_vac_m / (eps_r * SIGMA_NM * 1e-9)


QSTAR = math.sqrt(bjerrum_sigma(1.0) * EPS_R * 1.0)


N_CHAINS = 400
N_MONO = 30
N_BEAD = N_CHAINS * N_MONO


T_MONO, T_CAT, T_ANI = 1, 2, 3
SIGMA_T = {T_MONO: 1.0, T_CAT: 0.4, T_ANI: 1.6}
CHARGE_T = {T_MONO: 0.0, T_CAT: +QSTAR, T_ANI: -QSTAR}
MASS_T = {T_MONO: 1.0, T_CAT: 1.0, T_ANI: 1.0}


EPSILON = 1.0
FENE_K, FENE_R0 = 30.0, 1.5
CUT_MM = 2.0
WCA = 2.0 ** (1.0 / 6.0)
S_SOLV = 4.33
RC_SOLV = 5.0


SOLV_SIGMA = "one"

DT = 0.005
TDAMP = 1.0


def sigma_ij(i, j):
    return 0.5 * (SIGMA_T[i] + SIGMA_T[j])


def lj_cut(i, j):

    if i == T_MONO and j == T_MONO:
        return CUT_MM
    return WCA * sigma_ij(i, j)


def is_solvated(i, j):

    return (i == T_MONO) != (j == T_MONO)


def solv_sigma(i, j, mode=None):

    return 1.0 if (mode or SOLV_SIGMA) == "one" else sigma_ij(i, j)


def n_pairs(conc):

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
