# Kernel basis of the neural functional at eps_r = 2 (2026-09-30)

Same data (82 training runs incl. p40/p41), validation, protocol, readout and three seeds as the _lin models; only the
radial basis changes.  learn_eps2_kern.sbatch; figs/kern_compare.py; figs/knet_kernels.py -> interpretation/knet_kernels.png.
(a) widest Gaussian 4 sigma (M = 10, same log spacing), (b) 8 sigma (M = 13), (c) learned kernels B_m(r)[1 + g(r)],
g a 1-32-32-8 perceptron of r / s_max (g = 0: the Gaussians; radial, so every structural property holds:
Hessian asymmetry 4e-16, Noether 3e-10).  Parameters 18,841 / 19,359 / 20,136 / 20,225.

| basis | held-out chi2 c = 0.01 0.02 0.04 0.06 0.08 | profile % | p27 chi2 | c = 0.05 chi2 / % | validation |
|---|---|---|---|---|---|
| Gaussians to 2 sigma | 0.67 0.65 0.97 0.89 1.13 | 9.3 3.1 1.3 1.0 1.0 | 8.2 | 0.76 / 1.00 | ~1.03 |
| (a) to 4 sigma | 0.66 0.63 1.14 0.92 1.03 | 8.0 3.2 1.4 1.0 0.8 | 77 | 0.83 / 1.05 | 0.99 |
| (b) to 8 sigma | 0.67 0.62 1.37 0.87 1.00 | 8.6 3.0 1.5 1.0 0.8 | 225 | 0.81 / 1.04 | 1.00 |
| (c) learned kernels | 0.68 0.63 1.13 0.89 1.22 | 9.0 3.2 1.4 1.0 0.9 | 18 | 0.78 / 0.98 | 1.03 |

Gamma(k_1), MD sigma (c = 0.01 0.02 0.04 0.05 0.06 0.08), c = 0.01 seeds:
- 2 sigma: -3.1 0.0 +0.5 +0.8 +0.3 -1.8; 0.18 0.23 0.26
- (a): +1.4 +1.4 +1.1 +1.3 +0.4 -2.3; 0.55 0.37 0.59
- (b): -1.0 +1.7 +1.0 +1.3 +0.5 -2.0; 0.41 0.38 0.27
- (c): -1.9 +1.7 +1.0 +1.2 +0.4 -2.0; 0.36 0.29 0.25
Linear response at k_1 (MD p40 0.29 +- 0.07, p41 0.42 +- 0.09): 0.23 / 0.51 / 0.36 / 0.30 (2 sigma, a, b, c).
Learned kernels: 1 + g_m(r) is smooth and rises with r (1.2 at r = 0 to 1.5-2.9 at three widths): fatter tails, i.e. a
longer range, no new structure; the pair kernels u_ab(r) are almost those of the Gaussian basis.

Reading: a longer range raises the c = 0.01 Gamma (towards and past MD), but the seed spread (0.27-0.59) is larger than
the MD error and validation cannot tell the bases apart (0.99-1.03): the force data do not fix this quantity (as the
p40/p41 test showed).  The price is extrapolation in drive strength (p27 chi2 8 -> 77 -> 225, learned kernels 18) and a
slightly worse c = 0.04 and c = 0.05.  An 8 sigma Gaussian spans +-24 sigma, the whole box, so (b) is box dependent.
No basis is better overall; the 2 sigma Gaussians stay the best compromise.

## Learned kernels: envelope width x activation (2026-09-30)
Widths 1, 2, 3, 4, 6 sigma (M = 6, 8, 9, 10, 12) x activations silu, gelu, tanh, softplus (Phi and g), 3 seeds each,
same data/protocol (figs/knet_scan_summary.py -> interpretation/knet_scan.json; names ..._val3_kn<w><act>_s<seed>,
2 sigma silu = ..._knet_s<seed>).  All 60 models trained without failure; every EL solve converged.
Selected rows (mean of 3 seeds; Gamma deviations in MD sigma at c = 0.01 0.02 0.04 0.05 0.06 0.08):
| config | validation | held-out | p27 | c=0.05 chi2 / L2 % | Gamma dev | Gamma(0.01) seeds |
|---|---|---|---|---|---|---|
| Gaussians 2 sigma (quoted form) | 1.030 | 0.858 | 8.2 | 0.757 / 1.00 | -3.1 0.0 +0.5 +0.8 +0.3 -1.8 | 0.18 0.23 0.26 |
| learned 3 sigma gelu (best validation) | 0.947 | 1.024 | 166 | 0.852 / 1.08 | +2.9 +4.5 +1.7 +1.3 +0.2 -2.9 | 0.44 0.85 0.50 |
| learned 4 sigma silu | 0.985 | 0.855 | 112 | 0.824 / 1.04 | +1.1 +2.5 +1.0 +1.3 +0.4 -2.4 | 0.55 0.27 0.64 |
| learned 2 sigma silu | 1.025 | 0.898 | 18 | 0.776 / 0.98 | -1.9 +1.7 +1.0 +1.2 +0.4 -2.0 | 0.36 0.29 0.25 |
| learned 1 sigma softplus | 1.038 | 0.855 | 4.6 | 0.715 / 0.91 | +0.4 +1.9 +1.4 +1.7 +0.8 -1.3 | 0.53 0.43 0.36 |
| learned 1 sigma silu | 1.061 | 0.883 | 5.0 | 0.743 / 0.94 | -0.7 +1.0 +1.0 +1.2 +0.5 -1.3 | 0.32 0.29 0.51 |
| tanh (any width) | 1.19-1.32 | 0.97-1.14 | 7-214 | 0.77-1.02 | up to +10.8 at c = 0.01 | 0.42-1.28 |
Trends: wider envelopes lower the validation residual a little but raise p27 tenfold or more and worsen c = 0.05;
gelu and above all tanh push Gamma at low c far above MD; silu and softplus behave alike.
Selection: the validation residual (15 runs, and the minimum over training, so biased towards flexible models) does
not track any independent test here: its best configuration is worse than the quoted form on held-out, p27, c = 0.05
and Gamma.  On the independent tests the 1 sigma learned kernels (softplus or silu) are the best on every one, but
choosing on them uses the test data, so the choice must be confirmed on a fresh system (eps_r = 7.5) before adoption.
Gamma(0.01) spans 0.18-1.28 over all configurations and seeds: still not fixed by the force data.

## Confirmation at eps_r = 7.5 (2026-10-01): see salt_in_polymer_eps75/KERNELS_CONFIRM.md
1 sigma softplus: held-out 0.611 vs 0.629, c = 0.05 0.607 vs 0.625, strongest well chi2 2.6 vs 24; Gamma ~10 % higher
(mean |dev| 1.08 vs 0.83 sigma, max 2.4).  Two of the three pre-set criteria met, Gamma narrowly not.
