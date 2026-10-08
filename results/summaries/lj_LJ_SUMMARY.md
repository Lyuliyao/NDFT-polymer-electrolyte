# LJ benchmark of the architectures (2026-09-30)

2026-10-02: paper model (learned kernels, kn1softplus) added: held-out chi2 3.66+-0.26 (median 0.67), profile 0.42%, rho 0.5 2.88 / 0.31%; quarter data 0.82% profile (2 seeds early-stopped at step ~1130). Code now = ion code + LJ fixes (old Gaussian-era copy in code_gauss_0930). figs/lj_arch_summary_kn.py -> runs/learn/interpretation/arch_summary_kn.json.

System (Cheng 2026's LJTS): lj/cut 2.5 shifted, T = 1.5 (T_c ~ 1.08), cubic L = 24 sigma, NVT (Nose-Hoover), two identical
labels (N/2 each, l_B = 0) so that learn/ runs unchanged.  Densities 0.1 0.2 0.4 0.6 0.7 trained, 0.5 never trained.
22 planar neutral potentials per density from tools/make_potential.py (fourier 8, mixed 4, gauss 6, long 4; at rho >= 0.6
wells <= 2.5 kT and Fourier parts <= 2 kT pp, against freezing); 1000 tau transient, 10^4 tau production, profiles and
force densities from LAMMPS ave/chunk (16 blocks merged to 8), stored in units of k_B T.  130 of 132 runs pass (two mixed
runs with 1.2 sigma modes fail the finite-difference YBG check, chi2 3-4.5).  Zero field 10^5 tau: Gamma(k_1) = 1/S(k_1)
= 0.734 0.634 1.130 2.171 4.517 8.830 (+-1 %).  Training exactly as for the ions (force matching, val3, 4000 steps,
3 seeds), n_ref = 0.2 for every network.  Code: code/learn (copy of the ion code + three LJ changes: k_B T units in the
profiles, no Kerker preconditioning without charges, a 1e-30 guard in lambda_min_2x2 against the exactly degenerate
labels).  Scripts: lj_*.sbatch, tools/lj_*.py, figs/lj_*.py.

| model (params) | held-out chi2 (median) | held-out L2 % | Gauss L2 % | rho=0.5 chi2 | rho=0.5 L2 % | EL |
|---|---|---|---|---|---|---|
| ours (18,841) | 7.8±1.0 (0.80) | 0.47 | 0.38 | 7.5 | 0.41 | 123/123 |
| c1win +-3 sigma (32,514) | 52±3 (1.50) | 0.72 | 0.98 | 53 | 0.58 | 123/123 |
| c1win +-6 sigma (47,874) | 231 | 1.80 | 2.78 | 332 | 2.42 | 123/123 |
| cace R 1.5 sigma (23,828) | 91±15 (2.19) | 0.82 | 1.29 | 34 | 0.55 | 123/123 |
| cace R 3 sigma (23,828) | 545 | 3.30 | 5.36 | 455 | 2.71 | 123/123 |
| V1 pair closure (24) | 2385 | 11.3 | 19.8 | 1613 | 6.0 | 36/41 |
Validation chose +-3 sigma and R 1.5 sigma.  Strongest well (rho 0.4 p14, contrast 67): chi2 58 / 842 / 1258, profile
0.75 / 3.2 / 4.8 % (ours / c1win / cace).  The one run where ours is worse: rho 0.7 p09 (mixed, 1.2 sigma modes at the
highest density), chi2 17 against 5.
Gamma(k_1): ours, c1win +-3 and cace R1.5 within 2-7 % of MD at every density; V1 flat (1.05-1.37 against 0.63-8.83).
S(k) of the total density at rho 0.5, rms % in k bands [0,2.1] [2.1,5.3] [5.3,7.2]: ours 2.1 7.0 8.1; c1win 3.7 2.5 28;
cace 2.3 5.9 24 (the label partials are meaningless here: the label-difference channel is never driven).
Structure: ours exact (Noether 2e-14); c1win net force up to 1.4 % (median 0.06 %), Hessian of the total density
asymmetric by 50 % (median), reflection up to 15 %, closed loops up to 1.5 %; cace integrable, Noether 1e-6.

Training-set size (held-out chi2 median / held-out L2 % / rho=0.5 L2 %):
| fraction (runs) | ours | c1win +-3 | cace R1.5 |
|---|---|---|---|
| 0.25 (20) | 0.84 / 0.88 / 1.03 | 7.3 / 2.11 / 2.03 | 1.49 / 2.70 / 2.21 |
| 0.5 (39) | 0.84 / 0.49 / 0.55 | 3.8 / 0.94 / 1.10 | 1.86 / 0.90 / 0.71 |
| 1 (74) | 0.80 / 0.47 / 0.41 | 1.50 / 0.72 / 0.58 | 2.19 / 0.82 / 0.55 |

Reading: on the simple fluid ours is the most accurate of the three at every data size (forces 7-12x, profiles 1.5-1.7x
with all data; with a quarter of the data about as accurate as the others with all of it).  The Cheng form works here
(every EL solution converges, profiles < 1 %), so its failure on the ions is a property of that system, not of the
implementation.
