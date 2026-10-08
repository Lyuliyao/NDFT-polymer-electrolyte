# Paper Handoff (PNAS, 2026-09-29)

Target: PNAS Research Article. Structure and word budget in `output/doc/manuscript_outline.md` (Abstract 200 words and
Significance 110 words drafted; Introduction drafted, about 390 words). Figure logic in `notes/figure_plan.md`; numbers in
`notes/result_summary.md`. The pre-PNAS outline is archived in `output/doc/archive/manuscript_outline_pre_pnas.md`.

## Ready To Draft
- Section: Abstract, Significance Statement, Introduction
  - Input artifact: `output/doc/manuscript_outline.md` (drafts), `notes/literature_notes.md` (citation keys)
  - Main takeaway: polymer electrolytes need a closure beyond mean field; learned cDFT exists for one-component fluids; here a structure-preserving functional for two charged species in a polymer, with a pair kernel as the contrast.
- Section: Results: A structure-preserving neural density functional + Table 1
  - Input artifact: `input/closure_note.md` (Props 1-4, assumptions A1-A4), `figures/structural_checks.json`, Claim 7; Fig. 1 (`figures/fig1/`)
  - Main takeaway: four properties hold for every parameter value, checked to 1e-9-1e-16 on the trained functional; stability is not built in and is penalised.
- Section: Results 1, "Density response at the noise level" + Fig. 2
  - Input artifact: `figures/fig2/` (caption, `fig2_metrics.json`); Claims 1 and 2
  - Main takeaway: chi^2 at the noise floor at every concentration (three seeds 0.629 +- 0.003), profiles 1-3%, PB 12-30%; the never-trained c = 0.05 at 0.623.
- Section: Results 2, "A pair kernel is not enough" + Fig. 3
  - Input artifact: `figures/fig3/` (caption, `fig3_metrics.json`, script `make_fig3.py`); Claims 4 and 8; `<research eps2>/SEEDS_SUMMARY.md`
  - Main takeaway: at eps_r = 7.5 the pair kernel fits one concentration but not the sequence (Gamma - 1 proportional to n); at eps_r = 2 it fails outright (chi^2 up to 22; Gamma - 1 changes sign, 0.37 to 11) while the functional, retrained, stays at the noise; p27 and c = 0.01 at eps_r = 2 are the stated edges.
- Section: Results 3, "Structure and thermodynamics that were never trained" + Fig. 4
  - Input artifact: `figures/fig4/`; Claim 3; Claim 6 for one or two sentences on the long-wavelength runs (SI S6, `figures/si/long_wavelength/`)
  - Main takeaway: S_ZZ 2.5-3.4%, S_NN 2.2-2.8% at c = 0.02-0.08 (c = 0.01 in SI S4); Stillinger-Lovett exact; Gamma in Fig. 3c; the response holds down to k_1.
- Section: Results 4, "Where the concentration dependence sits" + Fig. 5
  - Input artifact: `figures/fig5/` (caption, `fig5_metrics.json`), `figures/kernel_k1.json`, `figures/kernel_seeds.json`; Claim 5
  - Main takeaway: Gamma - 1 = (n/2) sum W(k_1); the pair kernel's sum is one number, MD's rises (and changes sign at eps_r = 2); the functional follows, the rise sitting mostly in the anion-anion channel at 0.5-1.5 sigma. The Coulomb cancellation is an SI consistency check.
- Section: Discussion
  - Input artifact: outline Discussion bullets; `notes/literature_notes.md` Secs 3-4
  - Main takeaway: positioning against the Avni cut-off kernel and neural cDFT, why the built-in structure matters, scope stated once, outlook to SDFT and conductivity.
- Section: Materials and Methods (main text part) and SI Appendix S1-S10
  - Input artifact: `README.md`, `learn/` docstrings, `notes/decision_log.md`, `input/closure_note.md`, the eps_r = 2 campaign notes in `result_summary.md`
  - Main takeaway: CG model at two eps_r, potentials and acceptance, force-matching loss and training, Anderson + Kerker Euler-Lagrange solver, W(k), S(k) and Gamma with MD error bars; details and proofs in the SI.


## Revised for the new model and the comparison (2026-10-01)
- main.tex: new Results subsection + Table 2 (comparison with the one-body network and the lattice free energy), Methods paragraph 'Other learned functionals', theory/Methods describe the learned kernels; all numbers from the kn1softplus models; 9 pages. SI 22 pages: S7 kernel choice, S8.4 linear response, S9 comparison (Tables S9-S13), S10 cut-off kernel, S11 held-out profiles. Abstract and Significance unchanged (no mention of the comparison; the abstract is at 248 words).
- Open: user to review the new subsection and Table 2; Gamma at eps 7.5 is now about 10% above MD (2.4 sigma at c = 0.06); seed 0 at eps 2 under-predicts the m = 1 amplitude at c = 0.01-0.02 by 22-30% (SI S6).

## Typeset (2026-09-29)
- Main text: `output/latex/main.tex` (PNAS 2023 class, `latexmk`-free build: pdflatex, bibtex, pdflatex x2), 9 pages including references, nothing compressed. Table 1 typeset; figure legends in the .tex (the `figures/fig*/caption.md` files are older). References in `output/latex/references.bib`; entries marked VERIFY there were added from memory. Placeholders: authors, affiliation, contributions, e-mail, acknowledgments, data availability, the solid-polymer-electrolyte review (Hallinan & Balsara 2013 inserted as a proposal).
- SI: `output/latex/si/si.tex` (2019 PNAS SI class; the 2023 class does not compile with the SI style), 19 pages, S1-S10, Tables S1-S7, Figs S1-S8, paper style.
- Still open: page length (compress legends if the user asks), PNAS figure sizes (17.8 cm, >= 6 pt), the pre-long-wavelength pair-closure numbers in Results.

## Drafted
- Discussion (515 words) and the main-text Materials and Methods (507 words), 2026-09-29. Word audit in the outline's 'Length budget and word audit': main text about 3830 words, legends 1693 words, estimate about 7.2 PNAS pages; legends are the first thing to trim.
- Results, all four subsections as prose (2026-09-29, about 1430 words; `output/doc/manuscript_outline.md`). They cite SI Appendix S4, S6, S7, S8, which are not yet written; figure panels are referred to in lower case (Fig. 2a) to match the figure files, PNAS uses upper case.
- Abstract (248 words), Significance (120 words), Introduction, and the section "A structure-preserving neural density functional" as prose (`output/doc/manuscript_outline.md`); SI Appendix S1 with proofs of all four built-in properties (`output/doc/si_appendix_draft.md`).

## Still Blocked
- Results, Discussion and Methods are bullet outlines; Abstract, Significance and Introduction are drafts.
- Bibliographic gaps: Tsamopoulos2023 volume and pages; Bui2026 journal DOI; the theory references of the closure note are cited from memory and must be verified.
- PNAS figure format not yet applied: 17.8 cm double-column (our figures are 183 mm) and a 6 pt minimum font (some annotations are 5-5.5 pt). Do this once the figure set is frozen.
- SI tables S1, S2 and the SI figures other than the long-wavelength one are not built.

## Next Writing Step
- Draft Results 1-4 and the Discussion to the word budget in the outline (skill chain: `scientific-writing` -> `write-scientific-manuscript`), then Materials and Methods; count words against the 6-page budget before deciding whether Fig. 5 stays in the main text.
