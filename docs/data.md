# Data and model conventions

The held-out identifiers at dielectric 2 are restored from the archived selected model's `heldout_runs`. Its training scope excludes `c0.08/p05` and `c0.08/p12`, which appear as PASS in the broader MD acceptance table but are absent from the reported fit. `excluded.json` records this distinction; the raw acceptance verdicts are preserved. This gives the paper's exact 82 training, 15 validation and 20 held-out runs.

## Systems

| System | Meaning |
| --- | --- |
| `eps75` | Polymer electrolyte, dielectric constant 7.5 |
| `eps2` | Polymer electrolyte, dielectric constant 2 |
| `lj1` | One-component truncated, shifted Lennard-Jones control |
| `bigbox` | Larger-domain transfer at concentration 0.04 |
| `field2d` | Fields varying in two spatial directions at concentration 0.04 |

The directory `pilot_T1.0_c0.04_one_both` is the production concentration-0.04 dataset at dielectric 7.5; its historical directory name does not mark it as an excluded pilot. All model selection is based on validation data. Alternate windows, stencils, fractions and initializations retained under `data/models/` support the supplementary comparisons.

## Processed files

`profiles.npz` stores the bin positions, densities, internal force densities, standard errors, external potentials, box length and potential specification. Species order is cation, anion; the one-component fluid stores its single species under the `cation` key. `Sk.npz` contains per-volume structure factors. `D.json` supplies the state point's concentration, box and particle counts. `heldout_split.json` / `heldout.json` preserve the original held-out identifiers.

Number and charge structure factors use `S_NN = <|rho_+ + rho_-|^2>/V` and `S_ZZ = <|rho_+ - rho_-|^2>/V`. The manuscript's finite-wavenumber factor is `Gamma(k1) = 2 nbar / S_NN(k1)`. Generic model metadata can also contain the zero-wavenumber limit; it is a different quantity.

Per-run acceptance diagnostics are retained with their measured values. A published model's Euler–Lagrange convergence status is retained as well, including the baseline failures used in the paper comparison.

## Canonical numerical tables

`results/tables/paper-results.json` lists exact selected names, configurations, source paths and regenerated statistics. It defines the paper's test scope explicitly: 20 held-out fields plus 22 fields at concentration 0.05. The stored per-run evaluations are preserved as provenance; older raw evaluations can include additional linear-response probes or concentration 0.12. Those are outside the paper's test domain and are excluded from canonical summaries. No stability claim extends beyond the stated domain.

`results/figures/generalization/data/` contains the archived profiles and nonplanar maps used for Figure 3, including MD uncertainties and selected predictions. `results/figures/paper/` contains the five final vector figures from the manuscript. Other generated figure outputs are derived products.

## Availability

Processed profiles, configurations and selected trained parameters are in this repository. Raw particle trajectories, simulation restarts and the full nonplanar prediction cache are stored separately. These inputs are required to repeat raw-data processing; a fresh clone can verify the numerical tables and reconstruct the archived-array generalization figure.
