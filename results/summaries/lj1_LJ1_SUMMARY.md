# LJ benchmark with one species (2026-10-05)

User request: the single-component LJ fluid must be treated as one species, not as two identical labels. The two-label
benchmark (salt_in_polymer_lj, LJ_SUMMARY.md) is superseded by this root for the paper; its data are reused.

Data: the same 132 field runs (LJTS r_c 2.5, T 1.5, L 24 sigma, densities 0.1 0.2 0.4 0.6 0.7 trained, 0.5 never
trained, 22 planar neutral potentials per density), the two LAMMPS labels merged per block (tools/lj_merge_labels.py:
n = n_1 + n_2, force densities summed, errors over the 8 merged blocks, legacy-diagonal convention as before). Same
held-out split, same PASS set (two-label verdicts; the merged YBG check would flag four more runs, see
acceptance.json merged_pass). Zero field: n_pairs = N, S_NN only. chi2 values are not comparable with the two-label
ones: the merged force errors are ~sqrt(2) smaller per row (two label rows -> one), so chi2 roughly doubles for the
same residual.

Code: learn/ with ModelConfig.n_species (1 here; the ion path is unchanged, 69 reference outputs bitwise equal).
Training as before: force matching + spectrum term (KB=2, mode k), val3 early stopping, three seeds, nested fractions,
n_ref = 0.4 (the one-component density scale). Jobs lj1_jobs.txt (18487081-18487122, 2-11 min each).

| model (params) | held-out chi2 (median) | held-out L2 % | Gauss L2 % | rho=0.5 chi2 | rho=0.5 L2 % | S(k) rho=0.5 % | EL |
|---|---|---|---|---|---|---|---|
| ours, learned kernels (18,861) | 20.5±19.1 (1.57) | 0.34 | 0.29 | 13.1 | 0.19 | 8.1 | 123/123 |
| ours, Gaussian kernels (17,801) | 58±49 (4.0) | 0.44 | 0.43 | 39.9 | 0.34 | 14.9 | 123/123 |
| c1win +-3 sigma (24,577) | 92±42 (2.0) | 0.53 | 0.77 | 99 | 0.32 | 15.0 | 122/123 |
| c1win +-6 sigma (32,257) | 542±53 (9.3) | 1.63 | 2.80 | 624 | 0.94 | 14.8 | 120/123 |
| cace R 1.5 sigma (19,426) | 131±27 (3.2) | 0.54 | 0.83 | 45 | 0.31 | 13.0 | 123/123 |
| cace R 3 sigma (19,426) | 995±250 (37) | 1.95 | 3.20 | 813 | 1.71 | 40.6 | 123/123 |
| V1 pair closure (8) | 5970 (478) | 11.8 | 20.1 | 4049 | 6.6 | 43.8 | 41/41 |
Validation (val chi2) chooses +-3 sigma (30 vs 285) and R 1.5 sigma (11 vs 198), as with two labels.
Per-seed held-out chi2 of ours: 10.2, 8.7, 42.6 (one seed with a large outlier run); the medians 1.57±1.39.

Gamma(k_1) = 1/S(k_1), deviation from MD in MD sigma: ours +0.3 -0.4 +2.6 +3.2 -0.9 +0.3 (every density within 3.2
sigma, i.e. 2-3 %); Gaussian form up to +5.4; c1win +-3 up to +6.3; cace R1.5 up to -4.5; V1 flat at 1.0-1.1.
Structure on the MD profiles: ours and cace exact (Noether 3e-14 / 3e-8, Jacobian symmetric to 1e-15, reflection
1e-13 / 4e-15, loops 2e-14 / 7e-10); c1win net force up to 4 %, Jacobian asymmetry 1.1 (median 0.6), reflection 13 %,
loops 0.8 %.

Training-set size (held-out chi2 (median) / held-out L2 % / rho=0.5 L2 %):
| fraction (runs) | ours | c1win +-3 | cace R1.5 |
|---|---|---|---|
| 0.25 (20) | 15.6 (0.81) / 0.33 / 0.19 | 719 (3.7) / 1.14 / 0.70 | 669 (2.6) / 1.30 / 1.16 |
| 0.5 (39) | 16.8 (2.1) / 0.39 / 0.26 | 126 (3.3) / 0.67 / 0.53 | 136 (2.1) / 0.54 / 0.35 |
| 1 (74) | 20.5 (1.6) / 0.34 / 0.19 | 92 (2.0) / 0.53 / 0.32 | 131 (3.2) / 0.54 / 0.31 |

Reading: with one species the ordering is the same as with two labels: ours is the most accurate on forces (4-6x in
the mean chi2, 1.3-2x in the median) and on profiles (1.6x), and with a quarter of the runs it is as accurate as with
all of them while the other two lose a factor 2-4. The paper (Table 1 and the simple-fluid paragraph) now carries
these numbers: 20.5 / 0.34 %, 92 / 0.53 %, 131 / 0.54 %, pair 5970 / 11.8 %; quarter data 0.33 / 1.1 / 1.3 %.
Scripts: figs/lj1_arch_summary.py (-> runs/learn/interpretation/arch_summary.json, arch_frac.json, arch_summary.txt),
paper/figures/make_tab_lj1.py (-> paper/figures/si/lj1_kbK2_metrics.json).
