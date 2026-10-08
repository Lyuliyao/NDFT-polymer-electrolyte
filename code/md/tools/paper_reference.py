"""Reference values read off the figures of the Tsamopoulos & Wang SI
(mz3c00757_si_001.pdf), for the acceptance checks of Section 7.

These are digitised BY EYE from vector plots, so treat them as +-5% targets,
not exact numbers.  Concentration convention: the paper's c_s = [Li+]/[EO] is
twice our c_LJ = N_+/N_bead.

  Fig S1a : system volume V(T) at p = 0, curves for c_s = 0.02 ... 0.14
  Fig S2  : self-diffusion coefficients D_-, D_+, D_COM at T* = 1.0
  Fig S4a : longest Rouse time tau_R ~ 2.3e3 tau at T* = 1.0, c_s = 0.1
  Fig S7  : conductivity sigma ~ 1e-4 q^2/(tau sigma eps) at T* = 1.0
"""

# ---- Fig S2, T* = 1.0 : D in sigma^2/tau ----------------------------------
# c_s ->  (D_anion, D_cation, D_polymer_COM)
DIFFUSION_T1 = {
    0.02: (1.32e-2, 0.55e-2, 0.22e-2),
    0.04: (0.96e-2, 0.46e-2, 0.175e-2),
    0.06: (0.78e-2, 0.38e-2, 0.145e-2),
    0.08: (0.61e-2, 0.31e-2, 0.110e-2),
    0.10: (0.49e-2, 0.25e-2, 0.090e-2),
    0.12: (0.42e-2, 0.22e-2, 0.070e-2),
    0.14: (0.31e-2, 0.17e-2, 0.055e-2),
    0.16: (0.235e-2, 0.13e-2, 0.045e-2),
    0.20: (0.18e-2, 0.10e-2, 0.030e-2),
}

# ---- Fig S1a, T* = 1.0 : total system volume in sigma^3 -------------------
VOLUME_T1 = {0.02: 14450.0, 0.04: 14480.0, 0.06: 14520.0, 0.08: 14700.0,
             0.10: 14870.0, 0.12: 15000.0, 0.14: 15250.0}

TAU_ROUSE_T1 = 2.3e3          # Fig S4a, c_s = 0.1
SIGMA_T1 = 1.5e-4             # Fig S7, order of magnitude, q^2/(tau sigma eps)

N_BEAD = 12000


def cs_from_clj(c_lj):
    return 2.0 * c_lj


def _interp(table, cs, idx=None):
    ks = sorted(table)
    if cs <= ks[0]:
        lo = hi = ks[0]
    elif cs >= ks[-1]:
        lo = hi = ks[-1]
    else:
        lo = max(k for k in ks if k <= cs)
        hi = min(k for k in ks if k >= cs)
    f = 0.0 if hi == lo else (cs - lo) / (hi - lo)
    a, b = table[lo], table[hi]
    if idx is not None:
        a, b = a[idx], b[idx]
    return a + f * (b - a)


def expected(c_lj):
    """Reference values for our c_LJ, with the derived box edge and densities."""
    cs = cs_from_clj(c_lj)
    n_ion = int(round(c_lj * N_BEAD))
    V = _interp(VOLUME_T1, cs)
    return dict(
        c_lj=c_lj, c_s=cs, n_ion_pairs=n_ion,
        V=V, L=V ** (1.0 / 3.0),
        rho_all=(N_BEAD + 2 * n_ion) / V, rho_mono=N_BEAD / V,
        D_anion=_interp(DIFFUSION_T1, cs, 0),
        D_cation=_interp(DIFFUSION_T1, cs, 1),
        D_com=_interp(DIFFUSION_T1, cs, 2),
    )


if __name__ == "__main__":
    import math
    print("paper reference at T* = 1.0 (digitised from the SI figures, +-5%)\n")
    print(f"{'c_LJ':>5} {'c_s':>5} {'pairs':>6} {'V':>8} {'L':>7} {'rho_all':>8} "
          f"{'D_+':>9} {'D_-':>9} {'D_s':>9} {'tau_8sig':>9} {'prod steps':>11}")
    for c in (0.02, 0.04, 0.06, 0.08):
        e = expected(c)
        Ds = min(e["D_cation"], e["D_anion"])
        tl = 64.0 / (4 * math.pi ** 2 * Ds)
        print(f"{c:5.2f} {e['c_s']:5.2f} {e['n_ion_pairs']:6d} {e['V']:8.0f} "
              f"{e['L']:7.3f} {e['rho_all']:8.4f} {e['D_cation']:9.2e} "
              f"{e['D_anion']:9.2e} {Ds:9.2e} {tl:9.0f} {100*tl/0.005:11.2e}")
    print("\nD_s = min(D_+, D_-) = D_cation everywhere: the small cation is the")
    print("slow ion because the solvation potential binds it to the chains.")
