# eps_r = 7.5 confirmation of the basis chosen at eps_r = 2 (2026-10-01)

Learned radial kernels B_m(r)[1 + g(r)], widest Gaussian 1 sigma (M = 6), softplus, chosen at eps_r = 2 on its test
data; trained here with the quoted command and splits (verified identical), three seeds (learn_eps75_kern.sbatch,
names ..._val3_kn1softplus_long_s{0,1,2}; figs/kern75_compare.py).  Criteria fixed beforehand: (i) held-out,
profile and c = 0.05 not worse beyond the seed spread, (ii) Gamma deviations not larger, (iii) not worse on the
strongest well (validation run c = 0.08 p01).

| | validation | held-out | c = 0.01 .. 0.08 | profile % | c=0.05 chi2 / % | c=0.08 p01 chi2 (anion %) |
|---|---|---|---|---|---|---|
| quoted (Gaussians 2 sigma, silu) | 2.633±0.096 | 0.629±0.003 | 0.565 0.580 0.596 0.610 0.793 | 1.45 | 0.625±0.010 / 0.95 | 24.0 (3.4) |
| learned 1 sigma softplus | 0.805±0.042 | 0.611±0.007 | 0.561 0.574 0.582 0.615 0.721 | 1.44 | 0.607±0.002 / 0.86 | 2.6 (1.7) |
Gamma(k_1) (MD sigma): quoted +0.8 -0.2 -0.2 -1.3 +0.8 -1.7; new +1.4 +1.1 +0.9 +0.1 +2.4 -0.6 (c = 0.01..0.08 incl. 0.05);
mean |dev| 0.83 vs 1.08, max 1.7 vs 2.4 (c = 0.06, where the MD value 5.65 lies below c = 0.05's 5.66).

Verdict: (i) met (better: held-out 0.611 vs 0.629, c = 0.05 0.607 vs 0.625 and 0.86 vs 0.95 %), (iii) met by a wide
margin (the paper's weak spot, chi2 24 -> 2.6), (ii) not met: the new Gamma is ~10 % higher everywhere, within 2.4 sigma.
Together with eps_r = 2 (p27 8.2 -> 4.6, c = 0.05 0.757 -> 0.715, every Gamma within 2 sigma instead of -3.1 at c = 0.01)
the gain on the strongest drives is replicated at both eps_r.
