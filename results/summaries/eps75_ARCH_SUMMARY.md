# Comparison of architectures, both dielectric constants (2026-09-30)

Same data, splits, force matching, analytic Coulomb part, stability penalty, AdamW 4000 steps and val3 early stopping as the
quoted V2; three seeds; only F_ex changes (learn/models.py):
- ours (V2): radial pair kernels + pointwise perceptron on 16 Gaussian-smoothed densities (18,841 parameters);
- c1win (Sammuller 2023; ions: Bui & Cox 2025): perceptron from the densities in a window of +-3 sigma (32,514) or +-6 sigma (47,874) directly to mu(z) = -c1(z); no free energy;
- cace (Cheng 2026, Equi-cDFT, Eq. 15): F = sum_g rho_g [a_loc(rho_g) + a_CACE(rho_g, B_g)], O_h-invariant Cartesian moments (|l| <= 3, nu <= 2, one constant radial channel) on a 0.5 sigma voxel lattice, stencil 3 voxels (R = 1.5 sigma, as published) or 6 voxels (R = 3 sigma); species channels (2 + 13 per species, 17 cross); 23,828.
Window/stencil chosen by validation chi2 only (c1win +-3 sigma at both eps_r; cace R 3 sigma at eps 2, R 1.5 sigma at eps 7.5).
Scripts: figs/arch_summary.py, figs/arch_extra_summary.py; jobs learn_arch.sbatch, submit_arch_frac.sh.

## Accuracy (3 seeds; eps2 held-out without p27)
| eps_r | model | held-out chi2 | held-out L2 % | c=0.05 chi2 | c=0.05 L2 % | S_ab(k) c=0.05 % | EL converged |
|---|---|---|---|---|---|---|---|
| 2 | ours | 0.86±0.04 | 3.23±0.03 | 0.76±0.04 | 1.00±0.06 | 8.9 | 126/126 |
| 2 | c1win +-3 | 0.84±0.02 | 2.76±0.05 | 0.64±0.01 | 0.84±0.00 | 11.5 | 126/126 |
| 2 | c1win +-6 | 1.03±0.05 | 3.13±0.04 | 0.68±0.01 | 0.91±0.04 | 24.2 | 125/126 |
| 2 | cace R1.5 | 1.69±0.12 | 35±45 | 1.01±0.02 | 85±140 | 26.8 | 105/126 |
| 2 | cace R3 | 2.20±0.30 | 38±43 | 1.22±0.06 | 26±38 | 31.7 | 94/126 |
| 7.5 | ours | 0.63±0.00 | 1.45±0.03 | 0.63±0.01 | 0.95±0.02 | 5.0 | 126/126 |
| 7.5 | c1win +-3 | 0.69±0.02 | 1.85±0.01 | 0.66±0.01 | 0.97±0.03 | 7.5 | 126/126 |
| 7.5 | c1win +-6 | 0.77±0.03 | 1.95±0.05 | 0.70±0.02 | 1.14±0.02 | 10.3 | 126/126 |
| 7.5 | cace R1.5 | 2.47±0.34 | 4.57±0.56 | 1.72±0.48 | 7.55±2.35 | 25.2 | 117/126 |
| 7.5 | cace R3 | 21.5±12.7 | 9.30±2.57 | 14.2±6.3 | 12.2±6.7 | 69.7 | 104/126 |

Edge cases: eps2 p27 (6.5 kT wells, beyond the trained 4.5 kT): chi2 / anion L2 = ours 8.2 / 2.5 %, c1win +-3 52.5 / 14.9 %,
cace 58-370 / 15-92 %.  eps7.5 strongest validation well (c = 0.08 p01, 100x contrast, seed 0): chi2 ours 23.2, c1win 2.8, cace 332.
Final parameters after 4000 steps instead of the best-validation checkpoint change none of this (ours 0.85 / 0.63, c1win 0.81 / 0.69,
cace 1.7-2.0 / 2.2-5.3 held-out chi2 at eps 2 / 7.5).

Gamma(k_1), deviation in MD sigma (c = 0.01 ... 0.08):
- eps 2: ours -3.1 0.0 +0.5 +0.8 +0.3 -1.8; c1win +-3 +2.3 +0.6 +1.3 +1.7 +0.6 -2.1; cace R1.5 +6.4 +2.0 -0.2 0.0 0.0 -2.0
- eps 7.5: ours +0.8 -0.2 -0.2 -1.3 +0.8 -1.7; c1win +-3 +3.2 +0.9 +0.8 0.0 +1.9 -1.7; cace R1.5 +1.1 +0.2 -0.6 -1.7 +0.4 -2.1
(full linear response; for c1win the Hermitian part of K(k_1) gives the same Gamma to < 1 %)

## Structure on the MD profiles (max over runs; median in parentheses), eps 2 / eps 7.5
| | ours | c1win +-3 sigma | cace R1.5 |
|---|---|---|---|
| net internal force, Noether | 1e-8 (6e-17) / 3e-8 | 9e-2 (1e-3) / 5e-2 (1e-3) | 1e-5 (2e-8) / 8e-4 (2e-8) |
| Jacobian asymmetry ||J-J^T||/||J|| | 3e-16 | 0.51 (0.2) / 0.71 (0.2) | 3e-16 |
| max |W+- - W-+| / max |W| | 4e-16 | 0.23 / 0.47 | 4e-16 |
| reflection | 2e-14 | 0.20 (0.02) / 0.13 (0.01) | 1e-14 |
| closed loop of mu.dn (path dependence of F) | 2e-14 | 1.6e-2 (2e-3) / 8.7e-3 (1e-3) | 2e-7 |
| Stillinger-Lovett |A-1| | 2e-10 | 1e-11 | 7e-12 |
(cace Noether: the bin-grid voxels are not smooth; perfect screening holds for every model through the analytic Coulomb part)

## Training-set size (best-validation checkpoint; c1win +-3, cace as chosen)
| eps_r | fraction (runs) | ours ho chi2 / c=0.05 chi2 | c1win | cace |
|---|---|---|---|---|
| 2 | 0.25 (22) | 1.22 / 1.34 | 2.28 / 5.01 | 12.3 / 15.8 |
| 2 | 0.5 (42) | 0.97 / 0.83 | 1.46 / 1.06 | 9.5 / 9.4 |
| 2 | 1 (80) | 0.86 / 0.76 | 0.84 / 0.64 | 2.2 / 1.2 |
| 7.5 | 0.25 (19) | 0.72 / 0.79 | 1.31 / 1.70 | 4.9 / 2.7 |
| 7.5 | 0.5 (37) | 0.65 / 0.67 | 0.88 / 0.77 | 3.2 / 1.9 |
| 7.5 | 1 (71) | 0.63 / 0.63 | 0.69 / 0.66 | 2.5 / 1.7 |
Profile L2 at fraction 0.25: ours 4.1 / 1.7 %, c1win 5.0 / 2.8 % (eps 2 / 7.5).

## Reading
- c1win (the one-body networks the introduction contrasts with): with all runs it is as accurate as ours (eps 2 profiles slightly
  better, eps 7.5 slightly worse); with a quarter of the runs its residual is twice ours and its c = 0.05 residual 2-4 times;
  beyond the trained drive (p27) its residual is six times ours, but on the strongest eps 7.5 validation well it is better.
  Its structure holds only as well as the fit: net force up to 5-9 % (median 0.1 %), Hessian asymmetric by 20-70 %,
  W+- != W-+ by 23-47 %, reflection broken by 1-20 %, the free energy path dependent by 0.1-2 %.
- cace (a free energy, so integrable and symmetric like ours) does not reach the noise level: residual 1.7-2.5 at best,
  profile errors 4.6-38 %, Euler-Lagrange failures on 7-25 % of the test runs, spurious short-wavelength oscillations,
  S(k) off by 25-70 %.  Not an early-stopping or data-size effect.  Likely cause (not ablated): one constant radial channel
  inside a hard 1.5-3 sigma sphere and no pair term, i.e. no radial resolution of the ion correlations.
- Not supported: "our structure is more accurate than a one-body network with plentiful data".

## LJ benchmark (see salt_in_polymer_lj/LJ_SUMMARY.md)
Same three architectures on Cheng's LJTS fluid (T 1.5, planar fields, same protocol): ours most accurate at every data size
(held-out chi2 7.8 vs 52 vs 91, profiles 0.47 vs 0.72 vs 0.82 %); the Cheng form works there (EL 100 %, profiles < 1 %).
S(k) by k band (physical channels, ions c = 0.05 S_NN [0,2.1] [2.1,5.3] [5.3,7.2]): eps2 ours 3.3 9.9 19, c1win 6.6 4.4 21,
cace 4.9 9.3 19; eps7.5 ours 6.0 3.1 11, c1win 3.2 3.2 19, cace 10.3 3.8 5.4 -- no architecture is consistently better in S(k).
