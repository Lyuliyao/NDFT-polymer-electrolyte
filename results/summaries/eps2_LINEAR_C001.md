# Linear-regime runs at c = 0.01, eps_r = 2 (2026-09-30)

p40-p43: V_N = A cos(k_m z + phi) on both ions, (m, A) = (1, 0.03), (1, 0.05), (2, 0.03), (2, 0.05) k_BT; 4e4 tau each.
Acceptance: p40 p41 PASS; p42 REVIEW (driven-mode halves 3.1 sigma), p43 REVIEW (anion current 3.4 sigma).
Script figs/linear_gamma.py -> runs/learn/interpretation/linear_gamma_c001.json.  Gamma_lin = 2 nbar / S_NN(k_m) from the
in-phase response (quadrature 0 in every run), errors with the integrated autocorrelation time (300-1000 tau).

| run | k | dn_N/2nbar | MD Gamma_lin | V2 s0 / s1 / s2 (EL) | V1 |
|---|---|---|---|---|---|
| p40 | k_1 | 0.10 | 0.29 +- 0.07 | 0.19 / 0.24 / 0.27 | 1.74 |
| p41 | k_1 | 0.12 | 0.42 +- 0.09 | 0.20 / 0.25 / 0.28 | 1.74 |
| p42 | k_2 | 0.055 | 0.55 +- 0.11 | 0.34 / 0.38 / 0.41 | 1.75 |
| p43 | k_2 | 0.094 | 0.53 +- 0.06 | 0.35 / 0.38 / 0.41 | 1.75 |
Zero field: Gamma(k_1) 0.42 +- 0.06, Gamma(k_2) 0.39 +- 0.02.  The models' EL response equals their Hessian value to <10 %.
Reading: linear (the two amplitudes agree); at k_1 the driven MD value (0.34 +- 0.05 combined) agrees with the zero
field, so MD Gamma(k_1) ~ 0.38 +- 0.04; V2 (0.24, seed range 0.18-0.27) is ~3 sigma low, a model deviation, not MD
statistics.  At k_2 driven 0.53 +- 0.06 vs zero field 0.39 +- 0.02 (2.1 sigma, both driven runs REVIEW).
Halves differ by 1.4-1.7 sigma (slow modes), so each estimate carries 20-30 % error.
Not used for training yet (p40/p41 would enter the c = 0.01 training set and change its validation draw).

## Retraining with p40, p41 (user request, 2026-09-30)
learn_eps2_lin.sbatch, names ..._val3_lin_s{0,1,2} and v1_..._lin; learn.data.validation_split keeps family "linear" in
training so the validation draw is the quoted one (verified identical; training 80 -> 82 runs, only p40 p41 added).
Result: no change.  Parameters move by <= 0.2 % (V2) and 0.6 % (V1); held-out chi2 per concentration, profile errors,
c = 0.05 (0.757 +- 0.038, 1.00 %) and Gamma(k_1) identical to 2-3 decimals; Gamma(0.01) seeds 0.18 / 0.23 / 0.26 (was
0.18 / 0.23 / 0.27); the EL response on p40-p43 unchanged (0.23, 0.24, 0.38, 0.38).  figs/lin_compare.py.
Why: the quoted model already fits p40/p41 at the noise level in force space (chi2/bin 0.52, residual power in the
driven k_1 mode 0.6-4 of ~120-140 per run and species), and the two runs are 1.7 % of the loss.  The forces of a weak
long-wavelength drive cannot tell Gamma = 0.23 from 0.38; only the density amplitude can (the force residual is
k times the chemical-potential residual).  Consistent with the training-signal comparison (local-mu balance, a
density-based loss, got Gamma(0.01) right).
