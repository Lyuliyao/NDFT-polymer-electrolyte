# Reproduce the paper

## 1. Verify numerical tables without dependencies

```sh
python scripts/reproduce.py verify
```

This independently recomputes profile-error summaries for 33 selected model/fraction combinations from per-run evaluations, checks that train, validation and held-out identifiers are disjoint, and checks that the single-permittivity views carry the parameters of the permittivity-transferable functional. It uses the canonical 22 transfer fields `p00–p21` for the ions. Model convergence failures reported in the paper remain in the evaluations; the comparison averages exclude unconverged or collapsed solutions exactly as the table captions specify. The convolutional baseline's entries describe final iterates.

## 2. Reconstruct figures

Install `requirements.txt` with Python 3.12, then run:

The archived checkpoints use NumPy 2's serialization paths; the pinned NumPy version supports loading these arrays.

```sh
python scripts/reproduce.py figure --figure 3
```

Figure 3 uses the included compact arrays and requires no model fitting. The other figures use processed profiles and trained checkpoints. The output remains under `results/figures/`; the original final vector figures are in `results/figures/paper/`.

| Manuscript figure | Reconstruction script |
| --- | --- |
| 1: functional and learning pipeline | `make_fig1.py` |
| 2: planar density profiles | `make_fig2.py` |
| 3: larger boxes and nonplanar maps | `make_generalization.py` |
| 4: structure factors, number response and kernels at k1 | `kernel_k1.py`, then `make_fig3.py` |
| 5: transfer across the Coulomb coupling | `make_fig6.py` |

The filenames of the last two scripts retain their original numbering; `reproduce.py figure --figure 4` runs both scripts in order. Numerical helpers write derived JSON beside the figures. Figures 1–4 use the permittivity-transferable functional through its single-permittivity views and the learned pair closure selected by validation; Figure 5 uses the five-permittivity functional, the leave-one-out functionals and the summaries in `data/predictions/epsx/`.

## 3. Train the reported form

```sh
python scripts/reproduce.py train --form transferable
python scripts/reproduce.py train --form transferable --leave-out 4
python scripts/reproduce.py train --system eps75 --form ours
python scripts/reproduce.py train --system eps75 --form pair
```

The first command trains the reported functional on all five dielectric constants; `--leave-out` trains a functional without one of them and scores it on that dielectric constant. The fixed-coupling forms are trained on one dielectric constant.

`scripts/prepare_workspace.py` creates local links from the distributed data layout to the experiment roots expected by the computational code. Reproduced fits receive the suffix `_reproduced` and are written beneath `work/runtime/`. Archived checkpoints remain the reference for published numbers. The launcher specifies the paper configuration rather than the generic exploration defaults.

| Setting | Paper configuration |
| --- | --- |
| Dielectric constants | 7.5, 5, 4, 3, 2 (Bjerrum lengths 7.96 to 29.8 sigma); the permittivity enters only through the Bjerrum length |
| Coupling dependence | Excess free energy F0 + lambda F1 + s(lambda) F2 in the coupling lambda (`--lb-mode poly`) |
| Ion training concentrations | 0.01, 0.02, 0.04, 0.06, 0.08; concentration 0.05 is untrained |
| Training / validation / held-out | 276 / 75 / 86 runs for the five-permittivity functional; 71 / 15 / 20 at dielectric 7.5 and 82 / 15 / 20 at dielectric 2 for the fixed-coupling forms |
| Kernel envelopes | 6 Gaussian envelopes with learnable widths, initialized geometrically from 0.25 to 4 sigma and bounded to 0.1–8 sigma; width learning rate 30 times the base rate |
| Kernel network | 1–32–32–6; softplus |
| Scalar readout | 12–128–128–1; softplus |
| Optimizer | Full-batch AdamW; learning rate 0.001 to 0.00001; weight decay 0.00001 |
| Duration | 12000 steps; validation every 20 steps; parameters kept at the lowest validation residual |
| Loss | Force matching with median error floor and the k^-2 Fourier term |
| Selected seed, five-permittivity functional | 0, fixed before training |
| Selected seed, fixed-coupling neural functional | 0 at dielectric 7.5; 2 at dielectric 2 (lowest validation residual) |
| Selected seed, learned pair closure | 2 at dielectric 7.5; 1 at dielectric 2 (lowest validation residual) |

The one-component LJ control has no Coulomb term, density scale 0.4 and training densities 0.1, 0.2, 0.4, 0.6 and 0.7. Density 0.5 is excluded from training. It uses the same learnable-width settings; its archived configurations and selected fractions are recorded in `results/tables/paper-results.json`.

The released loss applies the correct single weight to the Nyquist endpoint on even grids, consistent with Parseval's identity and the linear pair-design matrix. The archived training implementation doubled that endpoint in its extra Fourier term. This correction affects new fits and recomputed loss values; archived parameters, predictions and reported profile errors are preserved.

The grid-resampling helper also handles the splitting and merging of Nyquist coefficients when changing an even grid. Archived grid-comparison summaries remain unchanged; rerunning this helper computes results with the corrected interpolation.

## 4. Evaluate a checkpoint

```sh
python scripts/reproduce.py evaluate --system eps75 --form transferable
```

The launcher copies the selected model into `work/evaluation/` before evaluating it. Regenerated metrics are kept there. Floating-point backends and library versions can produce small differences; rerun training is a new fit, while the released checkpoints and numerical tables define the reported result.

## 5. Reprocess molecular dynamics

See [the MD guide](../code/md/README.md) for physical parameters and processing commands. Raw trajectories are needed for rebinning, regenerating covariance-aware force errors, or recomputing all nonplanar predictions. The existing density arrays and published numerical results are preserved. The corrected 2D processor estimates internal-force uncertainty from paired total-minus-external force blocks; archived force-error maps have not been regenerated without raw trajectories.
