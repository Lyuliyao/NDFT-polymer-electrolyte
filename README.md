# A structure-preserving neural density functional for the ions of a polymer electrolyte

Code, data and results behind the manuscript *A structure-preserving neural density functional for the ions
of a polymer electrolyte* (Liyao Lyu, 2026; LaTeX source in `paper/`, synced from the Overleaf repository
`Lyuliyao/CDFT-electrolytes`, commit in `paper/SOURCE_COMMIT.txt`).

The functional is a classical density functional for the cation and anion densities of a coarse-grained
salt-doped polymer melt (Tsamopoulos and Wang model). The Coulomb mean field is analytic; the correlations
beyond it are a learned scalar functional built from six learned radial kernels and a pointwise perceptron,
so integrability, invariance under continuous translations, rotations and reflections, the Noether force and
torque identities and perfect screening hold by construction. It is trained by force matching on
molecular-dynamics (LAMMPS) density and internal-force-density profiles under static planar external
potentials, at two dielectric constants (eps_r = 7.5 and 2) and five salt concentrations (c = 0.01 to 0.08
ion pairs per monomer), and tested on held-out runs, an untrained concentration (c = 0.05), longer boxes,
two-dimensional potentials, bulk structure factors and the long-wavelength number response.

## Layout

| path | content |
|---|---|
| `code/learn/` | the JAX package: functional forms (`models.py`: ours, pair closure, one-body network, lattice free energy, convolutional free energy), data loading (`data.py`), force-matching training (`train.py`), evaluation, Euler-Lagrange solver and bulk structure (`evaluate.py`), the experiment driver (`protocol.py`), 2D solver (`field2d.py`) |
| `code/md/` | the MD campaign: system build, external potentials, profile analysis, acceptance tests (`tools/`), LAMMPS input templates (`templates/`), SLURM scripts (`slurm/`), the stage driver `run_state.sh`, big-box and 2D job scripts (`bigbox/`, `field2d/`), model documentation (`README.md`, `doc/`), LAMMPS unit tests (`tests/`) |
| `code/analysis/<system>/` | per-system training job scripts (`*.sbatch`) and summary/comparison scripts (`figs/*.py` of the research roots: architecture and signal comparisons, big-box and 2D summaries, long-mode relaxation fits) |
| `code/figures/` | the paper's figure and table scripts (`make_fig*.py`, `make_tab_*.py`, `figstyle.py`, `grid_resample.py`, ...) |
| `data/md/<system>/` | MD outputs that the training and the tests use: per field run `profiles.npz` (densities, force densities, their errors, the potential on the bins), `acceptance.json`, `analysis.txt`; per state `zero_field/` (structure factors `Sk.npz`, `Sk_grid.npz`, diffusion `D.json`), `zerofield_rep/` (replica Fourier modes for the pooled Gamma at eps_r = 7.5), `potentials/` (the external potentials, JSON and LAMMPS form), `field_analysis/` (acceptance summaries), `heldout_split.json` (the held-out tags) |
| `data/models/<system>/` | every trained model of the study: `config.json`, `params.pkl`, `metrics.json` (held-out force residuals, profile errors, bulk structure), `heldout.npz` (Euler-Lagrange profiles of the held-out runs), `predict_c005/` or `predict_c05/` (the untrained concentration); `interpretation/` holds the per-system summary JSON (architecture comparison, MD Gamma files, loss ablation) |
| `data/predictions/` | the models' predictions in the 2x, 4x, 8x boxes (`bigbox/<box>/<label>/predict_c004/`) with the big-box summaries, and the 2D density maps (`field2d/<system>/p??_pred.npz`) with the 2D summaries |
| `results/figures/` | the figures of the paper and SI (PDF/PNG/SVG) and the JSON files with every number behind them (`fig?_metrics.json`, `single_model_metrics.json`, `structure_metrics.json`, `kernel_k1.json`, `si/*.json`, `summaries_kbK2/`) |
| `results/notes/` | `decision_log.md` (dated log of every analysis decision), `result_summary.md`, `project_truth.md`, `closure_note.md` (the structural properties and their proofs), literature notes |
| `results/summaries/` | the per-system summary notes of the research roots |
| `paper/` | manuscript (`v1/submission.tex`, PNAS), arXiv version (`v1/arxiv.tex`), SI (`v1/si/`), figures |

Systems: `eps75` (eps_r = 7.5, the main system), `eps2` (eps_r = 2, strong coupling), `lj1` (the one-component
truncated and shifted Lennard-Jones fluid of the architecture comparison), `lj` (the earlier two-label LJ
benchmark, superseded by `lj1`), `bigbox` (c = 0.04 melts replicated 2, 4 and 8 times along z), `field2d`
(two-dimensional potentials at c = 0.04). `data/models/eps75_early/` holds the first eps_r = 7.5 models
(Gaussian-basis form, depth scans); the paper's eps_r = 7.5 models are in `data/models/eps75/`.

## The models reported in the paper

One trained model per form (the initialization with the lowest validation residual), in `data/models/<system>/`:

| | eps_r = 7.5 | eps_r = 2 | Lennard-Jones (`lj1`) |
|---|---|---|---|
| neural functional (19,641 parameters) | `joint_v2_L1_C4_R128x128_H64_<T>_val3_kn1softplus_kbK2_long_s0` | `joint_v2_L1_C4_R128x128_H64_<T>_val3_kn1softplus_kbK2_s1` | `..._<TL>_val3_kn1softplus_kbK2_lj1_s1` |
| pair closure, same learned kernels (1,336) | `joint_v1_L1_C4_R128x128_H64_<T>_val3_kn1softplus_kbK2_long_s2` | `joint_v1_L1_C4_R128x128_H64_<T>_val3_kn1softplus_kbK2_s0` | `..._lj1_s2` |
| one-body network, +-3 sigma (32,514) | `joint_c1win_W30_L1_C4_R128x128_H64_<T>_val3_kbK2_long_s2` | `joint_c1win_W30_..._val3_kbK2_full_s1` | `..._lj1_s2` |
| lattice free energy, R = 1.5 sigma (23,828) | `joint_cace_a5q3_L1_C4_R128x128_H64_<T>_val3_kbK2_long_s2` | `joint_cace_a5q3_..._val3_kbK2_full_s2` | `..._lj1_s2` |
| convolutional free energy (24,321) | `joint_cnn_d2_L1_C4_<T>_val3_force_kbK2_long_s1` | `joint_cnn_d2_L1_C4_<T>_val3_force_kbK2_s2` | `..._lj1_s1` |

with `<T>` = `c001_c002_c004_c006_c008` and `<TL>` = `c01_c02_c04_c06_c07`. The quarter- and half-data fits carry
`_f025` / `_f050`; the loss ablation without the Fourier term is `..._kn1softplus_long_s0` / `..._kn1softplus_s2`; the
window and stencil alternatives are `_W60` and `_a5q6`; the training-signal comparison uses `_lmu` and `_pcm`. The
selection and every number quoted in the paper are in `results/figures/single_model_metrics.json`.

## Reproducing

* Training and evaluation (CPU, a few minutes per model): `code/learn/protocol.py`; the exact command lines are the
  `*.sbatch` files in `code/analysis/<system>/` (e.g. `learn_kboost.sbatch` for the paper's functional,
  `pairkn.sbatch` in `code/md/slurm/` for the pair closure, `learn_arch.sbatch` for the baselines). A model root
  is a directory with `runs/<state>/field_p??/profiles.npz` and `runs/<state>/zero_field/Sk.npz`, as in
  `data/md/<system>/`; set `SIP_ROOT` to it.
* Figures and tables: `code/figures/make_fig*.py`, `make_tab_single.py` (SI tables), `grid_resample.py` (grid test);
  `figstyle.py` names the reported models. They read the model directories and the metric JSON files of
  `results/figures/`.
* MD: `code/md/README.md` documents the model and the stages (`run_state.sh equil | zerofield | potentials | field |
  analyze-field`); `code/md/doc/training_spec.md` the data specification.

Environment: Python 3.12, JAX 0.4 (CPU), optax, numpy, scipy, matplotlib; LAMMPS 2Aug2023 with PPPM for the MD.

## What is not included

The raw trajectories and restart files (4.3 TB: ion dumps every 200 steps of about 450 field runs, the zero-field
runs and their replicas, the big-box and 2D runs) stay on the MSU HPCC file systems
(`/mnt/gs21/scratch/lyuliyao/salt_in_polymer/{runs,eps2,bigbox,field2d}`); every quantity used in the paper is
derived from them by `code/md/tools/analyze_profiles.py`, `analyze_profiles2d.py`, `analyze_zerofield.py`,
`zf_modes.py` and `zf_pool.py`, whose outputs are in `data/md/`. The GitHub repository omits, in addition, the
models not quoted in the paper, the big-box and 2D prediction caches and the replica Fourier modes (see
`.gitignore`); the full archive (`NDFT_release_<date>.zip`) has everything listed above.
