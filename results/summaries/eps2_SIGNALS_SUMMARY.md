# Training-signal comparison, both dielectric constants (2026-09-30)

Same functional and architecture as the quoted V2 (joint L1 C4 R128x128 val3, five training concentrations, c = 0.05 never
trained, 3 seeds, 4000-step cap, early stopping on each signal's own validation loss); only the data term changes:
- force   = force matching on the YBG internal force density (ours, the quoted models);
- lmu     = local chemical-potential balance, Cheng 2026 Eq. 8: spatial variance of ln n + V + e phi + mu_theta, profiles and V only, unweighted;
- pcm     = pair-correlation matching, Dijkman 2025: model S_ab(k) from W(k) + Coulomb vs MD Sk_grid.npz (zero-field runs only, k = 2 pi m/L, m = 1..27).
Code: code/learn/signals.py, protocol.py --loss; tools/sk_partial_grid.py; summary figs/signals_summary.py (SIP_ROOT=<root>).
eps2 held-out excludes c = 0.04 p27 (as in SEEDS_SUMMARY.md). S_ab(k) % = rms relative error over k and the three partials at c = 0.05.

| eps_r | signal | held-out chi2 | held-out L2 % | Gauss wells L2 % | c=0.05 chi2 | c=0.05 L2 % | S_ab(k) c=0.05 % |
|---|---|---|---|---|---|---|---|
| 7.5 | V1 (ref.) | 1.89 | 3.71 | 4.08 | 1.42 | 1.67 | 4.15 |
| 7.5 | force | 0.63±0.00 | 1.45±0.03 | 0.58±0.06 | 0.63±0.01 | 0.95±0.02 | 5.05±0.26 |
| 7.5 | lmu | 0.66±0.01 | 1.51±0.04 | 0.61±0.10 | 0.67±0.01 | 0.90±0.01 | 4.50±0.05 |
| 7.5 | pcm | 122.7±16.5 | 5.15±0.21 | 2.09±0.66 | 112.3±3.7 | 7.20±0.01 | 4.08±0.82 |
| 2 | V1 (ref.) | 10.75 | 8.16 | 12.96 | 2.96 | 2.14 | 9.16 |
| 2 | force | 0.86±0.04 | 3.23±0.03 | 2.03±0.15 | 0.76±0.04 | 1.00±0.06 | 8.95±1.19 |
| 2 | lmu | 0.95±0.03 | 2.67±0.02 | 1.08±0.06 | 0.92±0.02 | 1.03±0.02 | 10.62±0.06 |
| 2 | pcm | 7.11±5.12 | 3.52±0.14 | 2.16±0.32 | 15.48±10.18 | 2.46±0.24 | 4.63±1.09 |

Gamma(k_1) = 2 nbar / S_NN(k_1), seed mean (deviation in MD sigma):

| c | 7.5 MD | force | lmu | pcm | 2 MD | force | lmu | pcm |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 1.14±0.16 | 1.28 (+0.8) | 0.91 (-1.4) | 1.08 (-0.4) | 0.42±0.06 | 0.23 (-3.1) | 0.35 (-1.1) | 0.46 (+0.6) |
| 0.02 | 1.97±0.26 | 1.91 (-0.2) | 2.05 (+0.3) | 1.91 (-0.2) | 0.98±0.13 | 0.99 (+0.0) | 0.85 (-1.1) | 0.94 (-0.4) |
| 0.04 | 4.05±0.63 | 3.89 (-0.2) | 4.24 (+0.3) | 4.18 (+0.2) | 3.14±0.47 | 3.39 (+0.5) | 3.38 (+0.5) | 3.37 (+0.5) |
| 0.05 | 5.66±0.57 | 4.93 (-1.3) | 5.30 (-0.6) | 5.15 (-0.9) | 4.41±0.49 | 4.79 (+0.8) | 4.98 (+1.2) | 4.82 (+0.8) |
| 0.06 | 5.65±0.54 | 6.06 (+0.8) | 6.43 (+1.5) | 6.37 (+1.3) | 6.09±1.06 | 6.40 (+0.3) | 6.68 (+0.6) | 6.51 (+0.4) |
| 0.08 | 9.74±0.75 | 8.47 (-1.7) | 9.01 (-1.0) | 8.86 (-1.2) | 11.05±0.82 | 9.56 (-1.8) | 10.30 (-0.9) | 9.76 (-1.6) |

Profile L2 % by stratum (held-out + c = 0.05 runs pooled, seed mean; strata from the force models' runs):

| stratum | 7.5 force | lmu | pcm | 2 force | lmu | pcm |
|---|---|---|---|---|---|---|
| kind = neutral (15 / 14 runs) | 0.93 | 0.94 | 2.54 | 2.36 | 1.91 | 2.60 |
| kind = charged (3 / 3) | 1.02 | 0.80 | 13.28 | 0.62 | 0.65 | 3.52 |
| kind = both (24 / 24) | 1.37 | 1.39 | 7.64 | 2.02 | 1.86 | 3.09 |
| weakest third (MD contrast) | 0.83 | 0.76 | 1.57 | 0.68 | 0.63 | 0.78 |
| strongest third | 1.37 | 1.39 | 13.11 | 4.11 | 3.58 | 5.32 |

Reading:
- pcm reproduces bulk S_ab(k) and Gamma (its target) and weak drives, but not strong or charge-separating drives:
  bulk states of an electrolyte are electroneutral, so pair correlations fix the functional's curvature only at n+ = n-
  (five points here); the nonlinear, charge-separated response is left to extrapolation. Worst at eps_r 7.5.
- lmu and force give the same accuracy at eps_r 7.5; at eps_r 2 lmu has lower profile error at trained concentrations
  (tie at c = 0.05) and a better c = 0.01 Gamma, force has lower force chi2. Consistent with the k-weighting: a force
  residual is n k x the mu residual, so force matching down-weights the longest wavelengths by k^2 relative to lmu.
- Same architecture gets Gamma(k_1, c = 0.01, eps_r 2) right when trained by lmu or pcm, so the -3.1 sigma of the
  quoted model is a training-weight issue, not a representational limit.
- Not supported: any claim that force matching is more accurate than the local-balance signal.
