# Molecular dynamics and profile processing

LAMMPS inputs and processing tools for the coarse-grained polymer electrolyte used in the paper. The simulation model follows Tsamopoulos and Wang, *ACS Macro Letters* **13**, 322 (2024), with ion–monomer solvation as specified below.

## Simulation model

All quantities use Lennard-Jones units: length `sigma`, energy `epsilon`, and time `tau`. The paper uses `k_B T = epsilon`.

| Quantity | Value |
| --- | --- |
| Polymer | 400 chains of 30 neutral monomers |
| Bond | FENE, `K = 30`, `R0 = 1.5` |
| Particle diameters | Monomer `1.0`, cation `0.4`, anion `1.6` |
| LJ interactions | `epsilon_ij = 1`; arithmetic diameter mixing |
| Nonbonded monomer cutoff | `2 sigma`, energy shifted |
| Other LJ cutoffs | WCA, `2^(1/6) sigma_ij`, energy shifted |
| Ion–monomer solvation | `-4.33 [(sigma/r)^4 - (sigma/5)^4]`, cutoff `5 sigma`; the same `sigma = 1` for both ions |
| Electrostatics | PPPM; dielectric constant `7.5` or `2` |
| Reduced ion charge | `±7.725213`; `l_B = 7.9572 sigma` at dielectric `7.5` |
| Integration | Time step `0.005 tau`; Nosé–Hoover thermostat with damping `1 tau` |
| State preparation | Equilibrate at zero pressure, then sample at the mean volume |
| Concentration | `c = N_+ / 12000 = N_- / 12000` |

`tools/model.py` defines the constants. `tools/write_ff.py` generates the force-field include; `templates/ff.lmp` is the dielectric-7.5 force field. Dielectric-2 runs require their own force-field include with `dielectric 2`, while retaining the same charges and other interactions. Supply the matching include and solvation table to the LAMMPS input for each system.

The solvation table generator emits `SIG1_MONO`, used by both production ion–monomer pairs, and two alternative-length sections. The force-field include selects the production section explicitly.

## Simulation stages

| Stage | Input / tool | Output |
| --- | --- | --- |
| Initial configuration | `tools/build_system.py` | LAMMPS data file |
| Soft push-off | `templates/in.01_pushoff` | Initial restart |
| Warm-up | `templates/in.02_warmup` | Warm restart |
| Zero-pressure equilibration | `templates/in.03_npt` | Mean-volume measurements and restart |
| Zero-field NVT | `templates/in.04_zerofield` | Ion trajectories and transport / bulk diagnostics |
| External fields | `templates/in.05_field` or `in.05_field_ck` | Ion coordinates, total forces, applied external forces |
| Planar profile processing | `tools/analyze_profiles.py` | `profiles.npz` |
| Two-dimensional profile processing | `tools/analyze_profiles2d.py` | `profiles2d.npz`, `acceptance.json` |

Use the inputs above with a local LAMMPS installation and supply the run-specific variables declared by each template. `env.sh` accepts overrides for `LMP`, `SIP_PY` and `SIP_ROOT`. Sampling durations and replica counts are defined in the manuscript; processed data and per-run metadata identify the released conditions.

`tools/make_schedule.py --mode production` generates 16 Fourier, mixed-frequency, and Gaussian-train potentials. `--mode long` generates six additional long-wavelength potentials; use `--tag-offset` to append them to an existing schedule. The paper's run-specific potential files and profiles are stored under `data/md/` at the repository root.

## Profile conventions

The density and force density are particle-count / force sums divided by the bin volume and number of sampled frames. Planar bins have width approximately `0.1 sigma`, adjusted to divide the box exactly. Eight consecutive time blocks give the profile standard errors. The internal force is the stored total force minus the stored applied external force; subtract these quantities within each block before estimating uncertainty.

The force-balance check uses `k_B T ∂_z n = f_tot`, equivalent to `k_B T ∂_z n = f_int - n ∂_z V`. The two-dimensional tool also retains per-mode time series for evaluating the modes resolved above the sampling noise.

From this directory, examples for an existing trajectory are:

```sh
python tools/analyze_profiles.py --run /path/to/field_p00 \
  --pot /path/to/potentials/p00.json --nblock 8 --mmax 40
python tools/analyze_profiles2d.py --run /path/to/field_p50 \
  --pot /path/to/potentials/p50.json --nblock 8
python tools/zf_modes.py --run /path/to/zerofield_rep/r00
python tools/zf_pool.py --state /path/to/state --orig /path/to/zero_field
```

The planar processor accepts `cat.dump` and `ani.dump` or their gzip versions. The two-dimensional processor expects the uncompressed dumps plus `current*.dat`. The zero-field mode processor expects an `ions.dump` with unwrapped coordinates.

## Bulk structure and uncertainty

`tools/zf_modes.py` evaluates microscopic species Fourier densities on integer box wavevectors. `tools/zf_pool.py` averages directions with equal wavevector magnitude and uses `S_NN = <|rho_+ + rho_-|^2>/V`, `S_ZZ = <|rho_+ - rho_-|^2>/V`, and `Gamma = 2 nbar/S_NN`. It estimates time-block and autocorrelation errors within a run and accounts for the spread across independent replicas. The published dielectric-7.5 bulk estimates use nine runs per concentration; dielectric-2 uses one run. Processed bulk estimates are included in the data release.

## Data scope

The repository distributes processed density and force profiles, external-potential specifications, bulk summaries, and model outputs. Raw LAMMPS trajectories, initial configurations, and binary restart files are not included in this Git checkout. Rebinning trajectories or regenerating their uncertainty estimates therefore requires the original raw files or new MD runs. `tests/` contains physical implementation checks for pair forces, electrostatics, applied external forces, and restarting simulations; running them requires LAMMPS and an appropriate initial configuration.

Reported convergence failures in the learned-functional comparisons are scientific results and remain part of the released evaluation outputs.
