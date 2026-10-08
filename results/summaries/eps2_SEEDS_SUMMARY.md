# Three-seed summary, both dielectric constants (2026-09-28)

V2 = joint L1 C4 R128x128 val3, five training concentrations (0.01 0.02 0.04 0.06 0.08), c = 0.05 never trained.
Seeds change only the parameter initialisation; data, held-out and validation splits are fixed.
Force chi2 = force-balance chi2 per bin on the MD density; L2 = EL profile relative L2, mean of cation/anion.
eps_r = 7.5 seed 0 reproduces the quoted `_long` model exactly (0.607 / 0.629 / 1.49 % / c=0.05 0.623).

| | model | train chi2 | held-out chi2 | held-out L2 % | c=0.05 chi2 | c=0.05 L2 % |
|---|---|---|---|---|---|---|
| eps_r 7.5 | V1 | 1.759 | 1.887 | 3.71 | 1.416 | 1.67 |
| eps_r 7.5 | V2 (3 seeds) | 0.605±0.002 | 0.629±0.003 | 1.45±0.03 | 0.625±0.010 | 0.95±0.02 |
| eps_r 2 | V1 | 9.288 | 14.467 (10.75 w/o p27) | 8.53 | 2.959 | 2.14 |
| eps_r 2 | V2 (3 seeds) | 0.775±0.051 | 1.227±0.332 (0.858±0.045 w/o p27) | 3.29±0.08 | 0.757±0.038 | 1.00±0.06 |

Gamma = 2 nbar / S_NN(k_1), the paper's definition (S from W(k) on the 8x fine grid with Coulomb, value at k_1;
`figs/gamma_k1.py`, identical to paper/figures/make_fig3.py; the k -> 0 value in metrics.json 'structure' is a different quantity):

| c | eps7.5 MD | V1 | V2 | eps2 MD | V1 | V2 |
|---|---|---|---|---|---|---|
| 0.01 | 1.14±0.16 | 1.98 | 1.28±0.07 (+0.8σ) | 0.42±0.06 (1e5 τ; 2e4 τ gave 0.37±0.08) | 1.74 | 0.23±0.05 (-3.1σ) |
| 0.02 | 1.97±0.26 | 2.95 | 1.91±0.06 (-0.2σ) | 0.98±0.13 | 2.49 | 0.99±0.10 (+0.0σ) |
| 0.04 | 4.05±0.63 | 4.84 | 3.89±0.16 (-0.2σ) | 3.14±0.47 | 3.94 | 3.39±0.09 (+0.5σ) |
| 0.05 | 5.66±0.57 | 5.75 | 4.93±0.11 (-1.3σ) | 4.41±0.49 | 4.64 | 4.79±0.06 (+0.8σ) |
| 0.06 | 5.65±0.54 | 6.63 | 6.06±0.10 (+0.8σ) | 6.09±1.06 | 5.32 | 6.40±0.15 (+0.3σ) |
| 0.08 | 9.74±0.75 | 8.28 | 8.47±0.05 (-1.7σ) | 11.05±0.82 | 6.60 | 9.56±0.26 (-1.8σ) |

Models: eps7.5 V1 = scratch runs/learn/v1_..._long, V2 = eps75/runs/learn/..._val3_long_s{0,1,2};
eps2 V1 = eps2/runs/learn/v1_..._full, V2 = eps2/runs/learn/..._val3_full{,_s1,_s2}.
