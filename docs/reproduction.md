# Reproduce the paper

## 1. Verify numerical tables without dependencies

```sh
python scripts/reproduce.py verify
```

This independently recomputes profile-error summaries for 31 selected model/fraction combinations from per-run evaluations and checks that train, validation and held-out identifiers are disjoint. It uses the canonical 22 transfer fields `p00–p21` for the ions. Model convergence failures reported in the paper remain in the evaluations; the comparison averages exclude unconverged or collapsed solutions exactly as the table captions specify. The convolutional baseline's entries describe final iterates.

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
| 4: structure factors and number response | `make_fig3.py` |
| 5: concentration-dependent kernels | `make_fig4.py` |

The filenames of the last two scripts retain their original numbering. Numerical helpers write derived JSON beside the figures. Kernel and structure calculations use the reported initialization, selected by validation.

## 3. Train the reported form

```sh
python scripts/reproduce.py train --system eps75 --form ours
python scripts/reproduce.py train --system eps2 --form ours
python scripts/reproduce.py train --system eps75 --form pair
```

`scripts/prepare_workspace.py` creates local links from the distributed data layout to the experiment roots expected by the computational code. Reproduced fits receive the suffix `_reproduced` and are written beneath `work/runtime/`. Archived checkpoints remain the reference for published numbers. The launcher specifies the paper configuration rather than the generic exploration defaults.

| Setting | Paper configuration |
| --- | --- |
| Ion training concentrations | 0.01, 0.02, 0.04, 0.06, 0.08 |
| Training / validation / held-out | 71 / 15 / 20 at dielectric 7.5; 82 / 15 / 20 at dielectric 2 |
| Kernel envelopes | 6 widths, geometrically spaced from 0.25 to 1 sigma |
| Kernel network | 1–32–32–6; softplus |
| Scalar readout | 12–128–128–1; softplus |
| Optimizer | Full-batch AdamW; learning rate 0.001 to 0.00001; weight decay 0.00001 |
| Duration | At most 4000 steps; validation every 20; patience 1000 steps after minimum 200 |
| Loss | Force matching with median error floor and the k^-2 Fourier term |
| Checkpoint selection | Lowest validation residual |
| Selected neural seed | 0 at dielectric 7.5; 1 at dielectric 2 |
| Selected learned-pair seed | 2 at dielectric 7.5; 0 at dielectric 2 |

The one-component LJ control has no Coulomb term, density scale 0.4 and training densities 0.1, 0.2, 0.4, 0.6 and 0.7. Density 0.5 is excluded from training. Its archived one-component configurations and selected fractions are recorded in `results/tables/paper-results.json`.

The released loss applies the correct single weight to the Nyquist endpoint on even grids, consistent with Parseval's identity and the linear pair-design matrix. The archived training implementation doubled that endpoint in its extra Fourier term. This correction affects new fits and recomputed loss values; archived parameters, predictions and reported profile errors are preserved.

The grid-resampling helper also handles the splitting and merging of Nyquist coefficients when changing an even grid. Archived grid-comparison summaries remain unchanged; rerunning this helper computes results with the corrected interpolation.

## 4. Evaluate a checkpoint

```sh
python scripts/reproduce.py evaluate --system eps75 --form ours
```

The launcher copies the selected model into `work/evaluation/` before evaluating it. Regenerated metrics are kept there. Floating-point backends and library versions can produce small differences; rerun training is a new fit, while the released checkpoints and numerical tables define the reported result.

## 5. Reprocess molecular dynamics

See [the MD guide](../code/md/README.md) for physical parameters and processing commands. Raw trajectories are needed for rebinning, regenerating covariance-aware force errors, or recomputing all nonplanar predictions. The existing density arrays and published numerical results are preserved. The corrected 2D processor estimates internal-force uncertainty from paired total-minus-external force blocks; archived force-error maps have not been regenerated without raw trajectories.
