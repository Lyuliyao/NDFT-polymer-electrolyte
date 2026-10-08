# Figure Plan (PNAS, 2026-09-29)

Five main figures and Table 1; everything else in the SI Appendix. One claim per figure, anchor panel named.
Scripts and outputs in `figures/` (`make_fig<N>.py` -> `fig<N>/fig<N>.{svg,pdf,png}` + `fig<N>/caption.md`); SI figures are
named by content (`make_figS_<name>.py` -> `si/<name>/`) and numbered only when the SI is assembled. Retired material
(old numbering) is kept in `figures/retired/`, never cited.

Renumbering from the pre-PNAS plan: old Fig. 7 (strong coupling) became Fig. 3 and absorbed old Fig. 3 (the eps_r = 7.5-only
nonlinearity figure, retired); old Fig. 4 lost its Gamma panel (now Fig. 3c); old Fig. 6 (long wavelength) moved to SI
Appendix S6. Figs 1, 2 and 5 are unchanged.

Standing rules: no results of models trained before the long-wavelength runs; no cross-training; no eps_r transfer test;
two named models ("the functional", "the pair closure").

## Fig. 1  Model and data pipeline (`make_fig1.py`)
- Claim: The excess functional is built from radial kernels, pointwise maps and an analytic Coulomb part, and is trained only on planar profiles.
- Panels: (a) MD box with external potential and a measured n_+(z), f_+^int(z) [methodological bridge]; (b) F_id | F_Coul | F_ex^theta on one level, the learned part "pair kernel + pointwise nonlinearity", mu, force matching [definition]; (c) what comes out: profiles, S(k), Gamma [definition].
- Anchor: (b). Explanatory text only in the caption.
- PNAS note: could shrink to 1.5-column width (114 mm) if the page budget is tight.

## Fig. 2  Density response at the noise level (`make_fig2.py`, `fig2_metrics.json`)
- Claim: On potentials never used in training, the functional predicts the ion density profiles to 1-3% at every concentration, where Poisson-Boltzmann is off by 12-30%.
- Panels: (a) three test profiles (c = 0.01 held-out, c = 0.05 never trained, c = 0.06 held-out), cation and anion [claim-supporting]; (b) force chi^2 per test run, 42 runs [validation]; (c) profile error per run [validation].
- Anchor: (b). SI: all held-out profiles (S10), per-concentration table (Table S1).

## Fig. 3  Pair closures miss the concentration dependence (`make_fig3.py`, `fig3_metrics.json`; former Fig. 7)
- Claim: The concentration dependence of the ion correlations needs the pointwise nonlinearity; at four times the Bjerrum length a density-independent pair kernel fails outright while the same functional, retrained, stays at the noise level and follows a Gamma whose excess changes sign.
- Layout: rows eps_r = 7.5 (top) and eps_r = 2 (bottom), shared axes per column; MD black, pair-kernel orange, functional blue with the three-seed range.
- Panels: (a) the same held-out potential (c = 0.02 p14) at both eps_r, cation density [case illustration]; (b) test chi^2 vs c, p27 at eps_r = 2 shown separately [ranking under a new regime]; (c) Gamma = 2n/S_NN(k_1) vs c with MD error bars [claim-supporting evidence].
- Anchor: (c).
- Legend keeps: how the pair closure is fitted; data counts at both eps_r; each eps_r trained separately; seed definition; Gamma - 1 proportional to n; p27 beyond the trained drive; the c = 0.01, eps_r = 2 limitation; MD error-bar method.
- SI: S7 without the pair term; S8 strong-coupling details (why eps_r, g(r), diffusion, S(k) at every c, per-run profiles, seeds, acceptance).

## Fig. 4  Emergent structure (`make_fig4.py`; `--all` -> SI `si/structure_all/`, `--eps 2` -> SI `si/structure_eps2/`)
- Claim: The charge and number structure factors, never used in training, follow from the second derivative of the functional.
- Panels: (a) S_ZZ(k)/2n, (b) S_NN(k)/2n at c = 0.02, 0.04, 0.05, 0.06, 0.08 [claim-supporting]. c = 0.01 is left out of the main figure (noisiest MD structure factors; its errors, 3.4% and 4.9%, are in the caption) and shown in `si/structure_all/` (user 2026-09-29). The eps_r = 2 SI version keeps c = 0.01, the state point where the functional misses.
- Numbers (rms over the plotted MD k-shells, seed 0): S_ZZ 2.5-3.4%, S_NN 2.2-2.8%. Gamma(c) is Fig. 3c.

## Fig. 5  Measuring the many-body part (`make_fig5.py`, numbers `kernel_k1.json`, `fig5_metrics.json`; rebuilt 2026-09-29)
- Claim: Gamma - 1 = (n/2) sum_ab W_ab(k_1); a pair kernel has one sum at every concentration, while MD's sum rises (and changes sign at eps_r = 2); the functional follows it, and the rise comes mostly from the anion-anion channel, then cation-cation, at r = 0.5-1.5 sigma.
- Layout: rows eps_r = 7.5 / 2, three columns; model colours as in Fig. 3, channel colours green (++), purple (--), grey-blue (+-).
- Panels: (a) sum_ab W_ab(k_1) vs c: MD as 2(Gamma - 1)/n, functional with seed band, pair kernel constant [claim-supporting, anchor]; (b) channel-resolved W_ab(k_1) vs c, functional solid, pair kernel dashed [claim-supporting]; (c) W_--(r), W_++(r) at c = 0.01 and 0.08, r = 0.3-2.5 sigma, seed bands [case illustration].
- Former real-space Fig. 5 (Coulomb cancellation) is SI S8: `make_figS_kernels.py` -> `si/kernels/` (eps_r = 7.5) and `si/kernels_eps2/`; the cancellation is the generic behaviour of a direct correlation function inside an excluded region, a consistency check.

## Table 1  Properties that hold by construction (main text, section "A structure-preserving neural density functional", before Results)
- Rows: E(3) covariance (Prop. 1; A1, A3) | integrability and the Ornstein-Zernike Hessian (Props 2a, 2b; A1, A3) | Noether force and torque sum rules, symmetric stress (Prop. 3; A1, A3, A4) | perfect-screening sum rules: electroneutrality and Stillinger-Lovett (Prop. 4 of SI S1.4; A2 and an invertible N^-1 + W(0)). Columns: property | assumption used | consequence | check on the trained functional (`figures/structural_checks.json`).
- Numbers: force sum rule <= 4.5e-9 (median 8e-17), translation equivariance 7e-13, W symmetry 4e-16, Stillinger-Lovett |A - 1| <= 2.2e-12, delta F / delta n vs finite differences 1e-10.

## SI Appendix (sections as in `output/doc/manuscript_outline.md`)
- eps_r = 2 SI figures drawn 2026-09-29 with `--eps 2`: `si/structure_eps2/` (as Fig. 4), `si/kernels_eps2/` (as Fig. 5), `si/long_wavelength_eps2/` (as the long-wavelength figure).
- S1 proofs of Props 1-4a and the 3D tests T0-T7; S2 structural checks; S3 MD and acceptance (both eps_r, eps_r = 2 data integrity); S4 learning details, seeds, Table S1 (per-concentration residuals and profile errors, all models, both eps_r), Table S2 (structure-factor errors); S5 Euler-Lagrange solver, W(k), S(k), Gamma and the MD error analysis; S6 long-wavelength response (`make_figS_long_wavelength.py` -> `si/long_wavelength/`, former Fig. 6) and the power spectrum of the training profiles; S7 ablations (learning curve, depth and width, without the pair term, packing term, mode weighting, early stopping); S8 strong coupling details; S9 the Avni cut-off kernel; S10 all held-out profiles at both eps_r.
