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

## Fig. 2  Density profiles at two coupling strengths (`make_fig2.py`, `fig2_metrics.json`; redrawn 2026-10-03)
Rows eps_r 7.5 / 2; columns c = 0.02 p14 (held-out), c = 0.05 p01 (never trained), c = 0.06 p14 (held-out); cation n/nbar;
MD band, Poisson-Boltzmann, pair closure, neural functional (seed 0), per-panel L2 errors. SI: `si/chi2_test` (chi2 per test run).

## Fig. 3  Bulk structure and thermodynamic factor (`make_fig3.py`, `fig3_metrics.json`; one column, 2026-10-03)
A S_ZZ, B S_NN at eps_r 7.5 (c = 0.02-0.08); C, D Gamma(c) at eps_r 7.5 and 2 (MD, NF band, pair closure).
SI: `si/chi2_conc` (test chi2 per concentration, the former Fig. 3A). SI structure figures: `make_figS_structure.py`
(`--all` -> `si/structure_all/`, `--eps 2` -> `si/structure_eps2/`, numbers `structure_metrics.json`).

## Fig. 4  Quantifying the many-body contribution (`make_fig4.py`, numbers `kernel_k1.json`, `fig4_metrics.json`; the former Fig. 5)
A combined kernel q^T W(k1) q vs c (MD estimate, NF, pair closure), B channels W_ab(k1), C real-space W_--(r), W_++(r).

SI size test: `make_figS_bigbox.py` -> `si/bigbox/` (driven-mode error vs box length), table `si_tab_bigbox.tex`.

## Table 1  Properties that hold by construction (main text, section "A structure-preserving neural density functional", before Results)
- Rows: E(3) covariance (Prop. 1; A1, A3) | integrability and the Ornstein-Zernike Hessian (Props 2a, 2b; A1, A3) | Noether force and torque sum rules, symmetric stress (Prop. 3; A1, A3, A4) | perfect-screening sum rules: electroneutrality and Stillinger-Lovett (Prop. 4 of SI S1.4; A2 and an invertible N^-1 + W(0)). Columns: property | assumption used | consequence | check on the trained functional (`figures/structural_checks.json`).
- Numbers: force sum rule <= 4.5e-9 (median 8e-17), translation equivariance 7e-13, W symmetry 4e-16, Stillinger-Lovett |A - 1| <= 2.2e-12, delta F / delta n vs finite differences 1e-10.

## SI Appendix (sections as in `output/doc/manuscript_outline.md`)
- eps_r = 2 SI figures drawn 2026-09-29 with `--eps 2`: `si/structure_eps2/` (as Fig. 4), `si/kernels_eps2/` (as Fig. 5), `si/long_wavelength_eps2/` (as the long-wavelength figure).
- S1 proofs of Props 1-4a and the 3D tests T0-T7; S2 structural checks; S3 MD and acceptance (both eps_r, eps_r = 2 data integrity); S4 learning details, seeds, Table S1 (per-concentration residuals and profile errors, all models, both eps_r), Table S2 (structure-factor errors); S5 Euler-Lagrange solver, W(k), S(k), Gamma and the MD error analysis; S6 long-wavelength response (`make_figS_long_wavelength.py` -> `si/long_wavelength/`, former Fig. 6) and the power spectrum of the training profiles; S7 ablations (learning curve, depth and width, without the pair term, packing term, mode weighting, early stopping); S8 strong coupling details; S9 the Avni cut-off kernel; S10 all held-out profiles at both eps_r.
