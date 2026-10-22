# Data and model conventions

The held-out identifiers at dielectric 2 are restored from the archived selected model's `heldout_runs`. Its training scope excludes `c0.08/p05` and `c0.08/p12`, which appear as PASS in the broader MD acceptance table but are absent from the reported fit. `excluded.json` records this distinction; the raw acceptance verdicts are preserved. This gives the paper's exact 82 training, 15 validation and 20 held-out runs at dielectric 2 for the fixed-coupling forms (71 / 15 / 20 at dielectric 7.5). The permittivity-transferable functional of the main text is trained on 276 runs and tested on 86 held-out runs across the five dielectric constants (see [reproduction.md](reproduction.md)).

## Systems

| System | Meaning |
| --- | --- |
| `eps75` | Polymer electrolyte, dielectric constant 7.5 (Bjerrum length 7.96 sigma) |
| `eps5`, `eps4`, `eps3` | Polymer electrolyte, dielectric constants 5, 4 and 3 (Bjerrum lengths 11.9, 14.9 and 19.9 sigma) |
| `eps2` | Polymer electrolyte, dielectric constant 2 (Bjerrum length 29.8 sigma) |
| `epsx` | Functionals trained across the five dielectric constants (models only; data are read from the five systems above) |
| `lj1` | One-component truncated, shifted Lennard-Jones control |
| `bigbox` | Larger-domain transfer at concentration 0.04 |
| `field2d` | Fields varying in two spatial directions at concentration 0.04 |

The directory `pilot_T1.0_c0.04_one_both` is the production concentration-0.04 dataset at dielectric 7.5; its historical directory name does not mark it as an excluded pilot. Fixed-coupling fits are selected by validation residual among three initializations. The permittivity-transferable functional is reported at initialization 0, fixed before training. Its single-permittivity views `data/models/eps75/FINAL_bwX_01` and `data/models/eps2/FINAL_bwX_01` carry the same parameters, evaluated on the runs of one dielectric constant. Alternate windows, stencils, fractions, initializations, left-out permittivities and conditioning forms retained under `data/models/` support the supplementary comparisons.

Model names encode the form (`v2`: neural functional; `v1`: learned pair closure), the training concentrations, `kn1softplus` (learned radial kernels, softplus activations), `kbK2` (the Fourier term of the loss) and `bw` (learnable envelope widths). In `epsx`, `E75-5-4-3-2` lists the training dielectric constants, `T4` marks a left-out dielectric constant and `lbpoly`, `lbinput` and `lbfilm` the coupling law, network-input conditioning and feature-wise modulation; a name without an `lb` tag has no coupling dependence.

## Processed files

`profiles.npz` stores the bin positions, densities, internal force densities, standard errors, external potentials, box length and potential specification. Species order is cation, anion; the one-component fluid stores its single species under the `cation` key. `Sk.npz` contains per-volume structure factors. `D.json` supplies the state point's concentration, box and particle counts. `coul_energy.json` holds the Ewald Coulomb energy of the zero-field run, used for the coupling-constant energy identity. `heldout_split.json` / `heldout.json` preserve the original held-out identifiers.

Number and charge structure factors use `S_NN = <|rho_+ + rho_-|^2>/V` and `S_ZZ = <|rho_+ - rho_-|^2>/V`. The manuscript's finite-wavenumber factor is `Gamma(k1) = 2 nbar / S_NN(k1)`. Generic model metadata can also contain the zero-wavenumber limit; it is a different quantity.

Per-run acceptance diagnostics are retained with their measured values. A published model's Euler–Lagrange convergence status is retained as well, including the baseline failures used in the paper comparison.

## Canonical numerical tables

`results/tables/paper-results.json` lists exact selected names, configurations, source paths and regenerated statistics. It defines the paper's test scope explicitly: at each of the two reported dielectric constants (7.5 and 2), 20 held-out fields plus 22 fields at concentration 0.05 (potentials `p00`–`p21`). The stored per-run evaluations are preserved as provenance; older raw evaluations can include additional linear-response probes or concentration 0.12. Those are outside the paper's test domain and are excluded from canonical summaries. No stability claim extends beyond the stated domain.

`results/figures/generalization/data/` contains the archived profiles and nonplanar maps used for Figure 3, including MD uncertainties and selected predictions. `results/figures/paper/` contains the five final vector figures from the manuscript. `data/predictions/epsx/` contains the bulk response and energy-identity summaries of the permittivity series used for Figure 5. Other generated figure outputs are derived products.

## Availability

Processed profiles, configurations and selected trained parameters are in this repository. Raw particle trajectories, simulation restarts and the full nonplanar prediction cache are stored separately. These inputs are required to repeat raw-data processing; a fresh clone can verify the numerical tables and reconstruct the archived-array generalization figure.
