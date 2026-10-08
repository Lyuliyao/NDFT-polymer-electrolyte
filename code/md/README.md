# CG-MD campaign: salt-doped polymer, statics-first training data

Implementation of `cg_md_setup.md` (companion to `sdft_learned_closure.md` §10)
on PSC Bridges-2 and MSU HPCC (amd20).  Model: Tsamopoulos & Wang, ACS Macro
Lett. **13**, 322 (2024), arXiv:2312.15401.  The solvation term uses sigma = 1
for both ion-monomer pairs since 2026-09-22 (see "Solvation screen").

## Layout

```
env.sh                 module loads, $LMP, $SIP_PY (Bridges-2; hands over to
env_msu.sh             ... this one on MSU HPCC)
build/lammps_bornsolv/ Hall's bornsolv LAMMPS (29Oct2020); build_amd20.sh
tools/
  model.py             all model constants; q*, l_B derived from CODATA
  build_system.py      initial melt + ions -> LAMMPS data file
  make_solv_table.py   1/r^4 solvation potential -> pair_style table file
  write_ff.py          generates templates/ff.lmp from model.py
  make_potential.py    one external-potential realisation -> JSON + LAMMPS include
  make_schedule.py     the 2 (pilot) / 16 (production) potentials per state point
  dumpio.py            LAMMPS dump / ave-time / ave-chunk readers
  analyze_zerofield.py density, D_+-, sigma/sigma_NE, S_ZZ(k), run-length table
  analyze_profiles.py  n_a(z), f^int_a(z), and the YBG check
templates/
  ff.lmp               GENERATED force field -- do not hand-edit
  in.01_pushoff        soft push-off
  in.02_warmup         full force field, capped Langevin -> NVT
  in.03_npt            NPT at p = 0  (fixes the density)
  in.04_zerofield      NVT zero-field production at the mean NPT volume
  in.05_field          NVT production with the external potential
tests/
  in.pairwrite         dumps every implemented pair potential
  check_pairwrite.py   compares them with the analytic model
slurm/                 equil / zerofield / field batch scripts (*_msu: MSU)
runs/                  one directory per state point
```

## The model as implemented

| item | value | where |
| --- | --- | --- |
| chains | 400 x N=30, FENE K=30, R0=1.5 | `model.py`, `build_system.py` |
| cation / anion | sigma 0.4 / 1.6, q = +-7.7252 | `model.py` |
| LJ | eps=1 all pairs, sigma_ij = (sigma_i+sigma_j)/2 | `ff.lmp` |
| cutoffs | mono-mono 2.0 (tail kept, unshifted); all ion pairs WCA at 2^(1/6) sigma_ij (shifted) | `ff.lmp` |
| solvation | ion-monomer only, S=4.33, **sigma=1** (both ions), rc=5.0, tabulated | `model.py` `SOLV_SIGMA`, `solvation.table` |
| electrostatics | `dielectric 7.5`, PPPM, real-space cutoff 5.0 | `ff.lmp` |
| ensemble | Nose-Hoover, Tdamp=1.0, dt=0.005 | `in.0*` |

Derived, and checked against the numbers quoted in the plan:

```
q*          = 7.7252          (plan: 7.73)
l_B(T*=1.0) = 7.957 sigma     (plan: 8.0)
l_B(T*=0.8) = 9.946 sigma     (plan: 10)
l_B(T*=0.6) = 13.262 sigma    (plan: 13)
U_coul(+- at 1.0 sigma) = -7.957 eps  (plan: about -8 k_B T)
```

### Pair-style layout

`hybrid/overlay` with four sub-styles:

* `lj/cut 1` -- monomer-monomer only, cutoff 2.0, **unshifted** (keeps the
  attractive tail the glass transition needs).
* `lj/cut 2` -- ion-monomer WCA core, **shifted** so U(rc)=0.
* `lj/cut/coul/long` -- ion-ion pairs, the only charged pairs in the system.
  Monomers carry q=0, so assigning Coulomb to ion pairs alone misses nothing
  and keeps the monomer-monomer neighbour list at cutoff 2.0 rather than 5.0.
* `table linear 100000` -- the ion-monomer 1/r^4 solvation term, overlaid on
  the WCA core.  Both ion-monomer pairs use the `SIG1_MONO` section (sigma = 1);
  `CAT_MONO`/`ANI_MONO` (sigma_ij) remain in the file for the old model.

Shifting changes no force and no virial, so it affects no measured quantity;
it is applied only so that reported energies mean what they say.

### Checked against the authors' Supporting Information

`paper/mz3c00757_si_001.pdf` (ACS SI for the paper) settles everything the
plan listed as unknown, and corrected one thing:

| SI statement | consequence here |
| --- | --- |
| eq. (1) is the **energy-shifted** LJ for every pair, monomer-monomer included | `pair_modify shift yes` globally (was: unshifted monomer-monomer). No force or virial changes. |
| ion-ion, ion-monomer and **bonded** monomer-monomer cut at 2^(1/6) sigma_ij | already correct: `bond_style fene` supplies the bonded WCA, `special_bonds fene` removes the 1-2 pair term |
| non-bonded monomer-monomer cut at 2.0 sigma | as implemented |
| S_ij = 4.33, r_c = 5.0 sigma, eps_r = 7.5, dt = 0.005 | as implemented; the length in eq. (3) is sigma = 1, not sigma_ij (settled by simulation, "Solvation screen") |
| Nose-Hoover thermostat **and barostat**, damping 1.0 tau, p = 0 | as implemented -- the aggressive-looking Pdamp is what the paper uses |
| Fig. S1a: V(T) at p=0 | gives the density the plan said was missing |
| Fig. S2: D_-, D_+, D_COM at T*=1.0 | gives the diffusion coefficients, hence the run lengths |
| Fig. S4a: tau_Rouse ~ 2.3e3 tau at T*=1.0 | equilibration budget |

Digitised into `tools/paper_reference.py` (by eye off vector plots, so +-5%),
and `analyze_zerofield.py --conc` now grades the pilot against them:

```
 c_LJ   c_s  pairs        V       L  rho_all       D_+       D_-  prod steps
 0.02  0.04    240    14480  24.374   0.8619  4.60e-03  9.60e-03   7.05e+06
 0.04  0.08    480    14700  24.497   0.8816  3.10e-03  6.10e-03   1.05e+07
 0.06  0.12    720    15000  24.662   0.8960  2.20e-03  4.20e-03   1.47e+07
 0.08  0.16    960    15250  24.798   0.9128  1.30e-03  2.35e-03   2.49e+07
```

D_s = min(D_+, D_-) = **D_cation** at every concentration: the small cation is
the slow ion, because the solvation potential binds it to the chains.  This
puts the whole campaign on the middle row of the plan's Section 6 table.

**One ambiguity the SI does not resolve.** S1 sets eps = k_B T at 400 K, so
T* = 1.0 <-> 400 K, which is what fixes q* = 7.7252 and l_B = 7.957 sigma.
But S2, after matching T_g to experiment, states that T* = 1.0 maps to ~450 K.
The two mappings are not consistent with each other; the charge must follow the
eps definition, so 400 K is used here (and is what the plan assumed).  Taking
450 K instead would give q* = 7.284, l_B = 7.07 sigma -- a 12% change in l_B.

### Validation

`tests/check_pairwrite.py` dumps every implemented pair potential with
`pair_write` and compares it with the analytic model:

```
mono-mono  energy 8.3e-15  force 1.0e-14   (of the curve's range)
cat-mono   energy 9.2e-08  force 5.8e-08
ani-mono   energy 2.7e-10  force 1.8e-10
cat-ani    energy 1.1e-09  force 1.1e-10
```

The residual is the linear interpolation of the 100 000-point solvation table.
`tests/run_all.sh` adds the dielectric-in-k-space and `fix addforce` checks; all
three pass on MSU with the 29Oct2020 build (2026-09-22).

## External potential

`make_potential.py` draws, for each part separately, 2-3 modes from
m in {3,4,5,6} (wavelengths 8 down to 4 sigma) with random phases and random
relative weights, then rescales so the peak-to-peak variation of that part is
the requested value in [0.5, 3] k_B T.  It emits

* a JSON spec (used by the analysis to reconstruct V_a and V_a'), and
* a LAMMPS include file with `v_fz_cat`/`v_fz_ani` (= -dV_a/dz) and
  `v_ez_cat`/`v_ez_ani` (= V_a), applied with `fix addforce`.

k_m = 2 pi m / L_z with integer m, so the potential is exactly box-periodic and
the wrapped coordinate LAMMPS gives an atom-style variable is the right one.
The potentials are generated **after** the box has been fixed at the mean NPT
volume, because L_z sets k_m.

## Output

Per field run, every 1 tau (200 steps), for the ions only:
id, type, position, total force, **and the external force LAMMPS applied**.
The last column makes f^int = f^tot - f^ext exact by construction rather than
dependent on the analysis reproducing the applied potential.  `fix ave/chunk`
also writes 0.1-sigma profiles on the fly in 500-tau blocks.

## Order of a state point

1. `slurm/equil.sbatch` -- build, push-off, warm-up, NPT at p=0.
2. `analyze_zerofield.py --npt ...` -- mean L, density, drift check.
3. `slurm/zerofield.sbatch LT=<mean L>` -- zero-field NVT production.
4. `analyze_zerofield.py --run ...` -- D_+-, sigma/sigma_NE, S_ZZ(k), and the
   run lengths implied by the measured D_s (Section 6 of the plan).
5. `make_schedule.py --lz <mean L>` -- the potentials.
6. `slurm/field.sbatch` as a job array over them.
7. `analyze_profiles.py` -- profiles and the YBG check.

`run_state.sh` drives these stages; on MSU it submits `slurm/<stage>_msu.sbatch`.


## Pilot result (hanhai20, 2026-09-21): the old model was too dense

With sigma_ij in the solvation term, the pilot reproduced the salt-free melt but
not the salt-doped one (LAMMPS 22Jul2025, converged NPT):

| | sigma_ij model | paper Fig. S1a |
| --- | --- | --- |
| salt-free melt, V | 14477 (rho = 0.829) | ~14420-14450 |
| c_LJ = 0.04, V | 13759 (rho = 0.942) | ~14700 |
| partial molar volume of salt | **-1.50 sigma^3/pair** | **+0.6 sigma^3/pair** |

The whole gap was the ion-monomer solvation, 83% of it from the anion
(sigma_ij^4 = 2.86 vs 0.24).  Equilibration, the thermostat, the table and the
400 K vs 450 K charge were ruled out.  The term comes from Hall's `bornsolv`
pair style, E = -S[(sigma/r)^4 - (sigma/r_c)^4], whose published inputs use
sigma = 1 throughout, which left four readings of eq. (3) to test.

## Solvation screen (MSU amd20, 2026-09-22): sigma = 1 for both ions

All four readings were run with `pair_style bornsolv` itself, from the
equilibrated hanhai20 state: 40 tau capped Langevin + 500 tau NVT, 500 tau NPT
at p = 0, 2000 tau NVT at <L>, c_LJ = 0.04, T* = 1.0
(`slurm/bornsolv_test_msu.sbatch`, results and `compare_comcorrected.py` in
`runs/bornsolv_test/`).

| | one_cation | ij_both (old) | **one_both** | ij_cation | paper SI |
| --- | --- | --- | --- | --- | --- |
| sigma in eq. (3) / ions | 1 / cation | sigma_ij / both | **1 / both** | sigma_ij / cation | |
| V (sigma^3) | 17812 | 13758 | **14709** | 20186 | 14700 |
| D_+ (sigma^2/tau) | 4.2e-3 | 3.9e-2 | **2.9e-3** | 9.3e-2 | 3.1e-3 |
| D_- (sigma^2/tau) | 2.8e-2 | 2.5e-3 | **5.2e-3** | 6.2e-2 | 6.1e-3 |
| D_- / D_+ | 6.6 | 0.07 | **1.79** | 0.67 | 1.97 |
| conductivity (q^2/(tau sigma eps)) | 7.9e-4 | 1.3e-3 | **1.3e-4** | 4.3e-3 | ~1.5e-4 |
| g_{+,mono} first peak | 15.3 | 5.2 | **12.2** | 6.2 | ~11.4-12.7 |

Only **sigma = 1 for both ions** reproduces the SI: V to 0.1%, D_+ and D_-
within 15%, D_-/D_+ within 10%, the conductivity, and the cation-monomer g(r).
The other readings fail qualitatively.  With sigma_ij the large anion carries
most of the solvation and becomes the slow ion (D_-/D_+ = 0.07).  Without
anion solvation the melt swells by 21-37% and its NPT volume keeps rising.  The
linear estimate that had favoured one_cation (V ~ 14718) missed by 3000
sigma^3, because the ion shells restructure.  With 400 K charges the match is
within the digitisation error, so the 400 K vs 450 K question needs no change.

**Production model since 2026-09-22:** `SOLV_SIGMA = "one"` in `model.py`, so
`write_ff.py` emits the `SIG1_MONO` table for both ion-monomer pairs.
`templates/ff.lmp` is byte-identical to `templates/ff_solv_one_both.lmp`, and
the old model is kept as `templates/ff_solv_ij_both.lmp`.  On one equilibrated
configuration the table and bornsolv agree to 7e-8 in E_pair, 1.4e-6 in P and
4e-6 of the rms force per atom, so the portable table is the bornsolv model.
Contact solvation energies are now -18.03 eps (cation-monomer, r = 0.70) and
-1.51 eps (anion-monomer, r = 1.30); they were -4.33 and -4.31.

### Centre-of-mass drift (fixed 2026-09-22)

`fix langevin` without `zero yes` gives the system COM a thermal momentum,
which the following Nose-Hoover stages then conserve.  In the first two screen
runs the whole system drifted at 0.010-0.015 sigma/tau.  That pushed the MSD
slopes to 1.3-1.9 and inflated D by 10-36x.  Since the fix, `in.01_pushoff`
and `in.02_warmup` use `zero yes`, and `in.04_zerofield` and `in.05_field`
begin with `velocity all zero linear`.  The measured drift is now below 1e-3
sigma/tau.  The screen's D values above are COM-corrected; treat any
zero-field D from before the fix as contaminated.

## Running on MSU HPCC

On MSU, `env.sh` hands over to `env_msu.sh`, which sets up:

* modules GCC 12.3, OpenMPI 4.1.5 and FFTW 3.3.10;
* `$LMP`, Hall's bornsolv LAMMPS 29Oct2020 built for amd20 by
  `build/lammps_bornsolv/build_amd20.sh` (KSPACE, MOLECULE, FFTW3,
  `-march=znver2`);
* `$SIP_PY`, the `mdmd` conda env.

`run_state.sh` then submits the `_msu` batch scripts (account multiscaleml,
constraint amd20, 32 ranks launched with `srun`).  Speed is about 550 steps/s
for 12 960 atoms on 32 ranks.  `compute reduce ... inputs local` only exists
from 21Nov2023 on, so `in.01_pushoff` picks the form via `$(version)`.  ICER
powertools has no `js`, so the scripts end with `sacct`.

## Converged state point: one_both, c_LJ = 0.04, T* = 1.0

`runs/pilot_T1.0_c0.04_one_both/`, built from scratch with the production
model and the default pipeline lengths (NPT 10 000 tau, zero-field 20 000 tau,
jobs 17575947 and 17596796).  This repeats the screen from a fresh melt rather
than from the hanhai20 configuration, so it is an independent check.

| quantity | measured | paper SI | ratio |
| --- | --- | --- | --- |
| L | 24.5023 +- 0.0002 | 24.497 | 1.000 |
| V | 14710.3 +- 1.6 | 14700 | 1.0007 |
| rho_all | 0.88102 | 0.88163 | 0.999 |
| D_+ | 2.95e-3 +- 6e-5 | 3.1e-3 | 0.95 |
| D_- | 5.85e-3 +- 8e-5 | 6.1e-3 | 0.96 |
| D_-/D_+ | 1.98 | 1.97 | 1.01 |

`analyze_zerofield.py --conc 0.04` grades this **PASS** on all four acceptance
quantities.  D and its error are the mean and standard error over four 5000-tau
blocks (`tools/convergence_report.py`); the single whole-trajectory fits are
D_+ = 2.80e-3 and D_- = 5.85e-3.

Convergence evidence:

* NPT, 10 000 tau: the two halves of the averaging window give 14711.2 and
  14709.3, the linear drift over the window is -4.6 sigma^3 (0.03%).
* The MSD log-log slope rises to the diffusive limit -- cation 0.93 over lags
  1000-3000 tau and 0.96 over 3000-6000 tau, anion 0.98 and 1.00.  The 2000-tau
  screen runs only reached 0.84-0.89, which is why D there was ~10% high.
* System COM drift 4.9e-5 sigma/tau, i.e. 200x smaller than before the fix.

**Not converged: the conductivity.**  The collective charge MSD is far noisier
than the single-ion ones.  Over the whole trajectory sigma/sigma_NE = 1.12 and
sigma = 3.2e-4 q^2/(tau sigma eps), against the ~1.5e-4 read off Fig. S7; but
the four blocks scatter from -4.7e-5 to 1.0e-3, so 20 000 tau and 480 ion pairs
do not pin this down.  sigma/sigma_NE near 1 is what the paper's model gives.
Quoting a conductivity needs longer runs or several seeds.

The final `write_restart` of the zero-field stage failed on the scratch **inode**
quota (1 048 576 files, user-wide), so the 20 000-tau trajectory has no restart
file; `runs/.../zerofield_tail/` is a short re-run (1000 tau equil + 1000 tau)
made only to give the field stage a starting configuration.

## Field stage (c_LJ = 0.04, T* = 1.0)

16 potentials, `potentials/p00..p15`: 12 with both parts, 2 with V_N only, 2
with psi only.  `make_schedule.py --mode pilot` returns the first two of that
list, so the pilot runs are production data rather than throwaways.  Both
pilot potentials have both parts active, because that is what shows the
charged part is applied with opposite sign to the two species.  Run lengths
come from the measured D_s: transient 1.2e6 steps (10 tau_lambda at 8 sigma),
production 1.16e7 steps (100 tau_lambda).

**PPPM accuracy: 1e-4 is enough, so the campaign does not need 1e-5.**
Section 7 of the plan asks for 1e-5 for the YBG check.  Measured against a
1e-6 reference on one configuration (`slurm/field_bench.sbatch`):

| | rms per-ion \|dF\| | binned \|dF_z\| / the YBG error bar |
| --- | --- | --- |
| pppm 1e-4 | 2.9e-4 | 3% (cation), 7% (anion) |
| pppm 1e-5 | 3.4e-5 | 0.4%, 0.7% |

The thermal rms force on an ion is 53.6 eps/sigma, so even treated as fully
systematic the 1e-4 k-space error stays well under the statistical error of
the profiles, while costing 2.1x less (422 vs 256 steps/s on 32 ranks).
p00 and p01 were run at 1e-5 before this was measured; they are kept.

**Scaling** (steps/s, whole node allocated, 12 960 atoms):

| ranks | pppm 1e-4 | pppm 1e-5 |
| --- | --- | --- |
| 32 | 422 | 256 |
| 64 | 667 | 368 |
| 128 | 815 | 435 |

64 ranks is the useful point.  Note that a field run sharing an amd20 node
with other jobs reached only 142 steps/s at 1e-5 on 32 ranks, against 256 on
an idle node: node sharing costs about 45%.

### Checkpointing, and running on the preemptible pool

A field run is 1.28e7 steps, several hours on any sane rank count, so the runs
checkpoint and resume rather than starting over when a job is cut short.

* `in.05_field` writes `restart.<step>` every `nrst` steps (default 5e5, about
  20 minutes), and ends with `run ${nsteps} upto`, which runs to an absolute
  timestep.  A resumed run therefore covers exactly what is left, and re-enters
  in the production phase instead of repeating the transient.
* `field_msu.sbatch` picks the newest checkpoint by the step in its name,
  deletes the older ones, and then **truncates the dumps back to that step**
  with `tools/dump_truncate.py`.  This is the part that is easy to get wrong:
  checkpoints are 2500 dump frames apart, so after a kill the dumps hold frames
  the checkpoint knows nothing about, and appending without cutting them first
  would duplicate those frames.
* Per-segment `prof_*.<seg>.dat` and `current.<seg>.dat`, because a restarted
  `fix ave/chunk` cannot append to the file of a previous attempt.  The YBG
  analysis reads the dumps, so it is unaffected.
* The script carries `--requeue` and `--open-mode=append`, so
  `PARTITION=scavenger ./run_state.sh field ...` survives preemption.

Verified on 2026-09-23 by letting a run be killed by its wall clock at step
17 400 with a checkpoint at 10 000, then resubmitting:

* the resumed job cut 38 stale frames from each dump and finished to 60 000;
* the merged dumps hold 301 frames, 0 to 60 000 in steps of 200, strictly
  increasing, no duplicates and no gap across the seam;
* a separate check ran 2000 steps straight through and 1000 + checkpoint +
  1000 from the same state: positions and velocities came out **identical**
  (max \|dx\| = 0, max \|dv\| = 0 over 960 ions), and the thermo at the common
  step matched digit for digit, so `fix nvt` carries its thermostat state
  through a restart and the seam is not a perturbation.

`tests/in.restartcheck` and `slurm/restartcheck.sbatch` reproduce that check.

### Pilot acceptance: PASSED (p00 and p01, full 58 000 tau)

`analyze_profiles.py` on the finished runs, plus the current log and the
on-the-fly profile blocks:

| | p00 (0.6 + 0.6 kT) | p01 (2.8 + 2.8 kT) |
| --- | --- | --- |
| applied force vs stored spec, per particle | 3.3e-5 / 3.5e-5 of range | 3.8e-5 / 3.9e-5 |
| YBG bins within 2 sigma (cation / anion) | 95.9% / 99.2% | 98.4% / 99.6% |
| chi^2 per bin | 0.84 / 0.61 | 0.70 / 0.50 |
| YBG driven modes, max \|diff\|/err | 1.21 / 1.83 | 1.66 / 0.76 |
| mean ion current, cation / anion | 0.74 / 0.92 sigma from zero | 0.06 / 0.12 |
| first vs second half, rms pull (cation / anion) | 1.27 / 1.19 | 1.20 / 0.95 |
| density contrast n_max/n_min | 2.11 / 1.50 | 105 / 12.2 |

All four acceptance checks of Section 7 hold: YBG bin by bin and mode by mode,
the halves agree, the mean ion current vanishes, and the zero-field run already
reproduced the paper.  The amplitude range is right at both ends: the weak
potential still gives a 2.1x density contrast, and the strong one is not
saturated (lowest cation bin 3.5% of the mean, no bin under 1%).

Two caveats on the error bars.  The half-to-half pulls have rms 0.95-1.27
rather than 1.0 because they use the scatter of 500-tau profile blocks, which
are correlated on the scale of tau_lambda = 579 tau; with that correlation
folded in the halves agree.  The YBG chi^2 per bin of 0.5-0.8 says the
eight-block error bars there are, if anything, conservative.

### A file edited under a running job

Both pilot jobs ended in `ERROR: Unknown command: ternal force LAMMPS applied`
*after* `write_restart`, with all 11.6e6 production steps done and the data
intact.  `templates/in.05_field` was edited while they were running: LAMMPS
reads its input by file offset, so the edit moved the offsets under the open
file and it resumed mid-comment.  The same thing hit `run_state.sh` earlier,
through bash, which reads scripts the same way.  A job that must survive edits
to the templates should copy its input into the run directory first and run
from the copy.


## The production campaign (2026-09-23/25)

Six state points with field data and one with zero field only, all at T* = 1.0,
PPPM 1e-4, on the preemptible pool with checkpointing.

| c_LJ | pairs | V | L | rho_all | rho_mono | D_+ | D_- | slope | tau_Fs(m=3) | transient | production | field runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 120 | 14454 | 24.359 | 0.847 | 0.830 | 4.57e-3 | 1.18e-2 | 0.959 | 269 | 1.6e6 | 7.4e6 | 16 |
| 0.02 | 240 | 14484 | 24.376 | 0.862 | 0.829 | 4.19e-3 | 9.45e-3 | 0.954 | 329 | 1.6e6 | 8.0e6 | 16 |
| 0.04 | 480 | 14710 | 24.502 | 0.881 | 0.816 | 2.80e-3 | 5.84e-3 | 0.947 | 518 | 1.2e6 | 11.6e6 | 24 |
| 0.05 | 600 | 14883 | 24.598 | 0.887 | 0.806 | 2.57e-3 | 4.66e-3 | 0.971 | 612 | 2.8e6 | 13.4e6 | 16 |
| 0.06 | 720 | 15089 | 24.711 | 0.891 | 0.795 | 2.02e-3 | 3.53e-3 | 0.934 | 791 | 3.6e6 | 17.2e6 | 16 |
| 0.08 | 960 | 15572 | 24.972 | 0.894 | 0.771 | 1.22e-3 | 2.29e-3 | 0.915 | 1302 | 5.8e6 | 28.6e6 | 16 |
| 0.12 | 1440 | 16750 | 25.586 | 0.888 | 0.716 | 4.27e-4 | 7.95e-4 | 0.880 | 3992 | - | - | 0 |

104 field runs, 547 GB of ion trajectories.  **c = 0.05 is held out** as an
interpolation test and **c = 0.12 has zero-field data only**, as an
extrapolation target: its field runs would have cost as much as the three
required state points together, for a state point the paper has no data for
and whose cation has not reached the diffusive regime even after 6e4 tau.

Each state point ran through `slurm/prod_chain_msu.sbatch`: build, push-off,
warm-up, NPT 1e4 tau, fix the box at <V>, NVT 2000 tau, zero field 2e4 tau
(6e4 for c >= 0.08), then the measurement, the 16 potentials and the field
array.  Every stage is skipped if its output exists, so a preemption costs one
stage.

### Run lengths are set from 1/(k^2 D_s), not from a measured tau_int

The plan asks for 50 tau_int(m=3).  The correlation time of the *collective*
amplitude cannot be measured well enough to set a run length: it is one complex
number per frame, so a 2-6e4 tau run holds only 30-50 independent samples.  The
estimates came out 3-14x below 1/(k^2 D_s), did not scale with D_s (222, 268,
316 tau while the physical time went 851, 1434, 4318) and ordered the modes
backwards (m = 6 slower than m = 2 at c = 0.12).

`tools/fs_relaxation.py` measures the same physics from the self part of
F(k,t), which averages over every ion and every time origin, so it carries
N_ion times the statistics.  It behaves: monotonic in m, monotonic in c, and
0.74-0.97 of 1/(k^2 D_s).  The production lengths use
tau_eff = 2 x 1/(k_3^2 D_s), the factor 2 being the in-field enhancement
measured at c = 0.04 (1.2-3.3 depending on mode); that reproduces the 58 000
tau which passed every test there.

### PPPM 1e-4 is enough, measured end to end

p00 and p01 at c = 0.04 were run twice, at 1e-5 and 1e-4, from the same
configuration and for the same 1.16e7 steps (`tools/compare_runs.py`):

| | worst driven-mode difference | n(z) chi^2/bin | f_int(z) chi^2/bin |
| --- | --- | --- | --- |
| p00 (weak) | 0.8 sigma | 1.41 / 1.33 | 1.23 / 1.19 |
| p01 (strong) | 1.5 sigma | 0.81 / 0.70 | 1.15 / 0.98 |

All 14 driven-mode amplitudes agree within 1.5 sigma with random signs.  This
confirms the static estimate made beforehand (the 1e-4 k-space error is 3-7% of
the YBG error bar) and buys a factor 2.1 in speed.

### The S_ZZ small-k check, extrapolated

R(k) = [S_ZZ(k)/(2n)] / (k^2/kappa_D^2) fitted as A + B k^2 over the six
smallest shells.  A = 1 is exact for any coupling: the k^2 coefficient is fixed
by the analytic Coulomb term alone, so this tests whether the simulation's
long-range force and the l_B used in the analysis are the same object.

| c_LJ | 0.01 | 0.02 | 0.04 | 0.05 | 0.06 | 0.08 | 0.12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1.005(27) | 1.053(21) | 1.001(26) | 0.982(20) | 0.984(15) | 0.965(13) | 0.990(6) |
| B | 0.69 | 1.01 | 1.42 | 1.57 | 1.52 | 1.56 | 1.37 |

The seven average to A = 0.997 and scatter in both directions, so there is no
common offset: dielectric, q*, the S_ZZ normalisation and the Ewald boundary
condition are consistent between simulation and analysis.  B > 0 is the
strong-coupling signature and is a target for the learned functional, not a
fault of the data.

## Acceptance of every field run (2026-09-25)

All 104 runs analysed by `slurm/analyze_field_msu.sbatch`.  **102 of 104 PASS.**

| c_LJ | PASS | driven-mode halves, median / worst | YBG bins within 2 sigma, median / min | mean current, worst |
| --- | --- | --- | --- | --- |
| 0.01 | 16/16 | 1.4 / 2.3 sigma | 98.0% / 96.3% | 1.9 sigma |
| 0.02 | 14/16 | 1.3 / 3.7 sigma | 98.2% / 95.1% | 1.2 sigma |
| 0.04 | 24/24 | 1.4 / 2.6 sigma | 98.0% / 91.8% | 1.7 sigma |
| 0.05 | 16/16 | 1.5 / 2.5 sigma | 97.6% / 95.1% | 1.1 sigma |
| 0.06 | 16/16 | 1.4 / 2.1 sigma | 98.0% / 95.1% | 1.8 sigma |
| 0.08 | 16/16 | 1.3 / 2.2 sigma | 98.8% / 93.6% | 1.8 sigma |

Over 208 driven-mode comparisons the pull has median 1.37 and rms 1.47, so if
anything the error bars are slightly tight; 2 exceed 3 sigma, both at c = 0.02
(p01 cation m = 3 at 3.0, p11 anion m = 16 at 3.7).  Their trend check shows no
systematic drift, so they are treated as fluctuations, not short runs.

### What the acceptance had to get right first

Five defects, each of which made healthy data look broken:

* **the box was hard-coded** to the c = 0.04 value, so every other state point
  was decomposed at the wrong k;
* **only m <= 7 was tested**, while a Gaussian train drives m = 4, 8, ..., 32
  and the mixed family reaches m = 20, so the driven modes were never checked
  and the undriven m = 1-3 did the flagging;
* **only the driven modes can gate.**  The undriven ones are thermal
  fluctuations whose slowest members hold ~10 independent samples per run; the
  same estimator that works on a driven mode underestimates their correlation
  time and produces 4-5 sigma pulls on healthy runs;
* **the mean-current error bar** was a plain block error, about twice too
  small, which flagged runs at 3-3.7 sigma that sit at 1.7-1.9 sigma once the
  correlation time is folded in;
* **the profile parser** assumed whole blocks, and a run preempted mid-block
  left a ragged tail that crashed the analysis (4 runs at c = 0.05).

With those fixed the flag rate fell from 8/56 to 2/104, and the pull
distribution became consistent with its own error bars.

## Learned functional (`learn/`, 2026-09-24)

`learn/` implements `doc/training_spec.md`: F = F_id + F_Coul + F_ex^theta, with
F_ex^theta learned from the accepted field runs by force matching
f^int_a = -n_a d_z (e_a phi[n] + mu^theta_a[n]).  JAX on the 1D periodic bin
grid, float64, CPU (`/mnt/home/lyuliyao/.conda/envs/heat/bin/python`,
jax 0.10 + optax; set `JAX_PLATFORMS=cpu`).

```
learn/geometry.py   grid, FFT convolutions, analytic Coulomb, Gaussian basis, folding
learn/data.py       accepted runs -> arrays; verdicts; splits/heldout.json; batches
learn/models.py     PB, Avni, V1 (pair), V2 (pair + net), V3 (net); mu = grad F; W(k)
learn/train.py      loss (spec §4), exact V1 least squares, AdamW + early stopping
learn/evaluate.py   Euler-Lagrange solver, profile metrics, S(k), Gamma, Stillinger-Lovett
learn/protocol.py   the §5 protocol as sub-commands; results in runs/learn/<name>/
learn/synthetic.py  the §7 checks:  python -m learn.synthetic
slurm/learn_msu.sbatch   sbatch slurm/learn_msu.sbatch full   (or any sub-command)
```

`python -m learn.protocol full` runs everything below in about an hour on four
cores; `python -m learn.protocol summary` tabulates `runs/learn/*/metrics.json`
into `runs/learn/summary.txt`.  Each experiment directory holds `config.json`,
`params.pkl`, `metrics.json` (per-run residuals, profile errors, structure at
every concentration), `heldout.npz` (predicted profiles) and `plots/`.

### Data actually used

A run counts when its verdict in `runs/field_analysis/acceptance_summary.csv`
is PASS (that file carries the YBG check; `acceptance.json` alone does not),
so 22/24 runs at c = 0.04 and 13/16 at c = 0.01 and 0.02.  Four held-out runs
per concentration, one per family (Fourier both / Fourier one-part / mixed /
Gaussian), drawn once with seed 20260925 into `splits/heldout.json`; the
learning curve is therefore 4 / 8 / 12 / 18 training runs, not 20.  Gaussian
trains are folded onto their period in Fourier space (harmonics of q only),
errors divided by sqrt(q).  Loss weights 1/(sigma_f^2 + eps^2) with eps the
median sigma_f of the run and species.  Zero-field S(k) exists at all seven
concentrations, so every model is scored against S_ab(k) at c = 0.01 ... 0.12
whether or not it saw that concentration.

### Where the specification had to change

* **Euler-Lagrange solver.** Picard iteration on the densities with mixing
  0.05-0.2 diverges: the mean-field Coulomb term gives the lowest box mode of
  rho_Z a linear gain kappa_D^2/k_1^2 = 25 (c = 0.01) to 100 (c = 0.04), so a
  stable Picard step would need mixing below 0.02.  `evaluate.el_solve` mixes
  the potentials u_a = V_a + mu^int_a (so n = N exp(-u)/Z stays positive) with
  Anderson acceleration over 8 steps and a Kerker factor k^2/(k^2 + kappa_D^2)
  on the charge channel.  Converges to 1e-6 in 15-70 iterations for every run
  of every model, including the 105:1 contrast of p01.
* **Stability criterion.** The exact V1 kernel, whose S(k) matches MD to 3%,
  has W_++(0) = -165, W_+-(0) = +125 at c = 0.04: nbar^-1 + W(k) alone is
  indefinite at small k in the charge channel, where 4 pi l_B/k^2 makes the
  real inverse structure matrix positive.  The penalty and the reported
  margin therefore use S^-1(k) = nbar^-1 + v_Coul(k) e e^T + W(k)
  (`models.inverse_structure`), on a k grid four (training) to eight (report)
  times finer than the box so that a pole between two box modes is not missed.
  The number channel is what remains at k = 0.
* **Bulk Hessian.** The response of mu^theta to a perturbation of the uniform
  state is linear, so one delta-function tangent per species (two JVPs) gives
  W_ab(k) at every k; the cosine-per-k version of §4 cost 0.4-0.75 s per step.
* Pair coefficients are O(100) kT sigma^3, so the trainable parameter is
  c/100 (`pair_scale`); the exact V1 least squares (condition number 9e3)
  needs no ridge (`v1ridge` selects 0 automatically under the criterion above).

### Synthetic checks (§7), all passing

delta F/delta n against central differences to 1e-10 for every variant
(including depth 3 and the gradient invariants); translation equivariance and
W_+- = W_-+ to 1e-15; V1 and Avni W(k) against their closed forms to 1e-15;
Stillinger-Lovett intercept A - 1 below 1e-9 for PB, Avni, V1 and a random
V3 (a 32x longer box supplies the small k).  A known V1 kernel is recovered
from six Euler-Lagrange profiles to 1e-11 without noise; with 1e-3 noise the
coefficients scatter by 6x (the eight widths overlap, condition number 3e5)
while u_ab(k) is still recovered to 2e-3 where the profiles carry signal.

### Results: c = 0.04 alone (18 training runs, 4 held-out)

Held-out chi^2 per bin (1 = at the MD error bar) and the Euler-Lagrange
prediction of the held-out profiles from V_a alone (relative L2, cation/anion):

| model | train chi^2 | held-out chi^2 | rel L2 | S_ZZ rel rms at 0.04 |
| --- | --- | --- | --- | --- |
| Poisson-Boltzmann | 411 | 233 | 0.27 / 0.33 | 0.77 |
| Avni cut-off kernel, a = 1 | 87 | 73 | 0.26 / 0.37 | 0.59 (unstable above c = 0.05) |
| V1, exact least squares | 0.60 | 0.74 | 0.013 / 0.013 | 0.032 |
| V3, L = 1 (learning curve n = 4 / 8 / 12 / 18) | 0.54 / 0.57 / 0.64 / 0.63 | 4.72 / 0.76 / 0.70 / 0.65 | 0.09 / 0.02 / 0.015 / 0.013 | 0.023 (n = 18) |
| V3 depth scan L,C = 1,4 / 2,4 / 2,8 / 3,4 / 3,8 | 0.63 / 0.65 / 0.64 / 0.64 / 0.64 | 0.65 / **0.61** / 0.62 / 0.66 / 0.63 | 0.013 / 0.012 / 0.013 / 0.013 / 0.014 | 0.02-0.05 |

Eight runs already bring the held-out residual to the noise floor; depth
gains 7% at best (L = 2, C = 4 is the chosen depth).  The learned pair kernel
alone predicts unseen profiles to 1% and S_ZZ(k), S_NN(k) at c = 0.04 to 3-4%
(`runs/learn/v1_c004/plots/`).  What single-concentration training does not
fix is the density dependence: every c = 0.04 model, V1 included, drifts away
from the MD S(k) at other concentrations, and the networks become unstable
(Gamma < 0) at c = 0.01 or 0.12.

### Results: joint training on c = 0.01, 0.02, 0.04 (36 runs, 12 held-out)

| model | held-out chi^2 | rel L2 | Gamma at 0.01 / 0.02 / 0.04 / 0.05 / 0.06 / 0.08 / 0.12 | S_ZZ rel rms, same order |
| --- | --- | --- | --- | --- |
| V2, L = 1 | 0.59 | 0.021 / 0.017 | 1.26 / 2.27 / 4.35 / 4.95 / 5.44 / 6.03 / 6.92 | 0.024 / 0.020 / 0.026 / 0.040 / 0.063 / 0.17 / 0.42 |
| V3, L = 1 | 0.60 | 0.024 / 0.019 | 0.98 / 1.69 / 3.20 / 3.01 / 2.52 / 1.39 / 0.33 | 0.035 / 0.020 / 0.024 / 0.054 / 0.20 / 0.46 / 0.63 |
| V1, exact least squares on the same 36 runs | 1.14 | 0.037 / 0.048 | 1.83 / 2.66 / 4.27 / 5.04 / 5.78 / 7.17 / 9.60 | 0.094 / 0.076 / 0.045 / 0.12 / 0.19 / 0.49 / 3.3 |
| MD, Gamma = 2n / S_NN(k_min) | | | 1.2 / 1.8 / 3.8 / 5.2 / 5.8 / 9.3 / 11.2 | |

A single density-independent kernel cannot fit the three concentrations
together: its residual doubles (1.14 vs 0.59), the anion residual of the
c = 0.01 and 0.02 Gaussian runs reaches 4.5, and Gamma - 1 comes out
proportional to nbar (1.83 at c = 0.01 against 1.2 from MD).  The density
dependence of the correlations is what the pointwise nonlinearity supplies.

One shared theta fits all three concentrations at the noise floor and solves
all 12 held-out profiles (`joint_v2_L1_C4_c001_c002_c004/plots/`).  With the
pair term (V2) the zero-field structure is predicted to 2-4% at c = 0.05 and
6% at 0.06 without any data there, S_ZZ still to 17% at 0.08, and Gamma
rises monotonically with c; the pure network (V3) reproduces the training
concentrations equally well but its Gamma turns over above c = 0.05 and its
S(k) is off by 20% already at 0.06.  S_NN at 0.08 and 0.12 is overestimated
by both (the S_NN(k -> 0) of MD keeps falling with c).

### Transfer

| trained on | predicts | model | chi^2 on all 13 PASS runs there | rel L2 of the predicted profiles | S_ZZ / S_NN rel rms there | Gamma there |
| --- | --- | --- | --- | --- | --- | --- |
| 0.02, 0.04 | 0.01 | V2, L = 1 | 0.60 | 0.042 / 0.034 | 0.024 / 0.041 | 1.26 |
| 0.02, 0.04 | 0.01 | V3, L = 1 | 0.65 | 0.096 / 0.063 | 0.088 / 0.33 | 0.41 |
| 0.01, 0.04 | 0.02 | V2, L = 1 | 0.59 | 0.019 / 0.017 | 0.033 / 0.037 | 2.32 |
| 0.01, 0.04 | 0.02 | V3, L = 1 | 0.98 | 0.027 / 0.029 | 0.028 / 0.075 | 2.16 |

Interpolation in concentration works for both; extrapolation down to
c = 0.01 needs the pair term (V3 without it puts Gamma at 0.41, i.e. close
to a spinodal, and its S_NN is 33% off).  The V2 model trained on
{0.02, 0.04} predicts the 13 profiles at c = 0.01 to 3-4% with a force
residual at the MD noise, and every Euler-Lagrange solve converged.

**Depth 2 at the joint level** (`joint_*_L2_C4_*`, `transfer_*_L2_C4_*`):
V2 with L = 2 gives the same held-out residual as L = 1 (0.597 vs 0.587)
and a worse extrapolation in concentration (S_ZZ 10% off at c = 0.05,
Gamma no longer monotonic); its transfer tests land at chi^2 0.59 / 0.69
and 3-5% profile error, the same as L = 1, while V3 with L = 2 has Gamma < 0
at c = 0.12 in both transfer directions.  The extra convolution layer buys
nothing on this data; V2 with L = 1 is the model to carry forward.  Not yet run:
`grad_invariants`, and the c = 0.06, 0.08 field data (their production runs
are still in the queue; add them with `--joint-conc 0.01 0.02 0.04 0.06 0.08`
once `analyze_profiles.py` and the acceptance have been run, keeping c = 0.05
out of training).

### Early stopping on a validation split (`--nval 3`)

The protocol stops on the held-out loss and then scores on the same runs.
Repeating the joint and transfer experiments with 3 runs per concentration
moved from training to a validation set (`*_val3`) gives:

| experiment | held-out chi^2, stop on test | stop on validation | transfer chi^2 / rel L2, test-stop -> val-stop |
| --- | --- | --- | --- |
| joint V2 L = 1 | 0.587 (36 train) | 0.614 (27 train) | |
| joint V3 L = 1 | 0.597 | 0.626 | |
| V2 {0.02, 0.04} -> 0.01 | 0.606 | 0.693 | 0.60 / 0.038 -> 0.69 / 0.045 |
| V3 {0.02, 0.04} -> 0.01 | 0.626 | 0.683 | 0.65 / 0.079 -> 0.69 / 0.068 |
| V2 {0.01, 0.04} -> 0.02 | 0.603 | 0.629 | 0.59 / 0.018 -> 0.64 / 0.021 |
| V3 {0.01, 0.04} -> 0.02 | 0.607 | 0.634 | 0.98 / 0.028 -> 1.28 / 0.030 |

The optimism of stopping on the test runs is worth 0.03-0.09 in chi^2,
part of which is the smaller training set.  Nothing above changes: V2 stays
at 0.61-0.69 against 1.14 for the joint V1 fit, the transferred profiles stay
at 2-5%, and S_ZZ and Gamma at the transfer target move by less than the
difference between models.  These are the numbers to quote.

### What the learned functional looks like (`learn/interpret.py`)

`python -m learn.interpret` writes `runs/learn/interpretation/`: the learned
pair kernels u_ab(r), the full bulk kernel W_ab(r) at several concentrations
(3D inverse transform of W(k)), and Gamma(c) as a continuous curve against
the MD estimate 2n/S_NN(k_min).

* The learned kernels have the opposite sign to the bare Coulomb interaction
  at short range: u_++ and u_-- are attractive wells of 20-60 kT at contact
  with range about 1 sigma, u_+- is a repulsion of 40-75 kT.  Added to
  l_B e e / r, the effective cation-anion potential is flat or slightly
  repulsive inside r = 1 sigma instead of the bare -40 kT: the monomer
  solvation shell around the cation (contact energy -18 kT) removes ion
  pairing, and like-ion pairs sharing a shell see an effective attraction.
  Below r ~ 0.5 sigma the curves are an extrapolation of the Gaussian basis,
  since the profiles carry no power beyond k ~ 5.
* The density dependence that the network adds sits almost entirely in the
  anion-anion channel: W_--(r) of joint V2 is a bare well at c = 0.01, grows
  a positive shoulder at r = 1.5-2 sigma at c = 0.04, and deepens to -45 kT
  sigma^3 with a longer range at c = 0.12, while W_++ and W_+- barely move.
  The large, weakly solvated anion is the species whose correlations change
  with crowding.  The split between the pair part and the network is not
  unique: V2's pair part is milder than the V1 kernel and the network carries
  the rest.
* Gamma(c): within the training range V2 (L = 1) follows the MD points and
  V1 is 20-50% too high at c = 0.01-0.02, because a density-independent
  kernel forces Gamma - 1 proportional to nbar.  Beyond the training range
  the picture reverses: V1's linear law reaches 9.6 at c = 0.12 against 11.2
  from MD, V2 saturates at 6.9, V3 collapses towards the ideal value and V2
  with L = 2 oscillates.  The nonlinearity is needed to get the density
  dependence right where there are data, and is not controlled where there
  are none: that is what the c = 0.06 and 0.08 runs are for.

### Predicting a new concentration, and adding one

`python -m learn.protocol predict --model runs/learn/<name> --conc 0.05`
scores a saved model on every PASS run at that concentration without training
(force residual, Euler-Lagrange profiles, S(k)); checked to reproduce the
transfer numbers on c = 0.02.  `--nval N` on `train`, `joint` and `transfer`
takes N runs per concentration out of the training set for early stopping,
so that the held-out runs are used for scoring only (names get `_valN`).
`splits/heldout.json` gains entries for a new concentration on first use,
with its own seed; existing entries are never changed.

Once the c = 0.05, 0.06, 0.08 field runs finish (each 16-35 M steps; on
2026-09-24 they were 15-50% through), the order is:

1. `sbatch --array=0-15%8 --export=ALL,STATEDIR=$PWD/runs/prod_T1.0_c0.06 slurm/analyze_field_msu.sbatch`
   (and c0.08, c0.05) -> `profiles.npz`, `acceptance.json` per run;
2. `$SIP_PY tools/acceptance_summary.py runs/pilot_T1.0_c0.04_one_both runs/prod_T1.0_c0.0{1,2,5,6,8} --csv runs/field_analysis/acceptance_summary.csv`
   -> refreshes the verdict table that `learn/` reads (all state points in
   one call, since the file is rewritten);
3. `python -m learn.protocol predict --model runs/learn/joint_v2_L1_C4_c001_c002_c004 --conc 0.05`
   -- the cleanest test, a concentration the model has never seen;
4. `python -m learn.protocol joint --nval 3 --variants v2 v3 --depth 1 --C 4 --joint-conc 0.01 0.02 0.04 0.06 0.08`
   then `predict ... --conc 0.05` with the new model, and the S(k) at 0.12
   is in its `metrics.json` under `structure`.

### First test on new data: c = 0.05 and 0.06 (2026-09-24, evening)

The 32 field runs at c = 0.05 and 0.06 finished and passed acceptance
(16/16 each; the refreshed table also turned the two REVIEW:current runs at
c = 0.04 p05 and c = 0.01 p03 into PASS, so those now train as well).  None
of the models below had seen either concentration.  Force residual chi^2
per bin (cation / anion) over all 16 runs, Euler-Lagrange profile error
(relative L2) and the mean relative error at the density minimum:

| model, trained on 0.01-0.04 | at 0.05: chi^2 | rel L2 | err at min | at 0.06: chi^2 | rel L2 | err at min |
| --- | --- | --- | --- | --- | --- | --- |
| V2, validation-stopped | 0.76 / 1.91 | 0.015 / 0.018 | 0.04 / 0.03 | 1.35 / 9.1 | 0.023 / 0.029 | 0.07 / 0.05 |
| V3, validation-stopped | 0.95 / 3.67 | 0.020 / 0.023 | 0.07 / 0.05 | 3.1 / 22.7 | 0.044 / 0.043 | 0.19 / 0.08 |
| V1, exact least squares | 1.09 / 3.95 | 0.030 / 0.031 | 0.07 / 0.05 | 2.3 / 11.1 | 0.046 / 0.045 | 0.10 / 0.09 |

Every Euler-Lagrange solve converged.  At c = 0.05 the V2 cation residual is
at the noise and the profiles are predicted to 1.5-2%; the anion residual is
elevated in the mixed and Gaussian runs (short wavelengths).  At c = 0.06,
past the training range, the cation still holds (1.35, 2.3%) while the anion
residual grows to 9 and the anion is over-depleted by up to 20% in the deep
minima of the strongest Gaussian well (`joint_v2_..._val3/predict_c005_c006/plots/selected.png`).
The species ordering is the one the real-space analysis predicted: the
anion-anion channel carries the density dependence.

**Retrained with c = 0.06** (`*_c001_c002_c004_c006_val3`, 38 training runs,
16 held-out over four concentrations; c = 0.05 stays entirely held out):

| model | held-out chi^2 | rel L2 | predicts c = 0.05: chi^2 | rel L2 | err at min | Gamma at 0.05 / 0.06 / 0.08 / 0.12 | S_ZZ rel rms, same order |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | 0.61 | 0.018 / 0.015 | 0.63 / 0.77 | 0.012 / 0.009 | 0.03 / 0.01 | 4.45 / 4.95 / 5.17 / 5.30 | 0.045 / 0.032 / 0.082 / 0.28 |
| V3 | 0.62 | 0.021 / 0.018 | 0.73 / 0.95 | 0.015 / 0.012 | 0.03 / 0.01 | 2.95 / 3.11 / 2.35 / 0.73 | 0.039 / 0.029 / 0.16 / 0.34 |
| V1 | 1.39 | 0.032 / 0.040 | 0.60 / 0.96 | 0.011 / 0.011 | 0.04 / 0.02 | 5.82 / 6.70 / 8.36 / 11.27 | 0.021 / 0.063 / 0.21 / 1.02 |
| MD, Gamma = 2n/S_NN(k_min) | | | | | | 5.2 / 5.8 / 9.3 / 11.2 | |

With both neighbours in the training set, c = 0.05 is predicted at the MD
noise by all three models, V1 included: an interpolation in concentration
does not discriminate them.  What does is the fit across concentrations
(V1's held-out residual grows from 1.14 to 1.39 as a fourth concentration is
added, V2 stays at 0.61) and the two ends of the range.  Above the training
range the picture is unchanged from the three-concentration fit: V2's Gamma
plateaus near 5 where MD rises to 9 and 11, V3 collapses, and the
density-independent V1 kernel happens to follow the MD trend up to c = 0.12
(11.3 vs 11.2) while its S(k) there is off by 100%.  The c = 0.08 runs (12 of
16 still running) are the ones that will decide whether the network can
learn the steep rise of Gamma; a possible architectural reason for the
plateau is that the lifted inputs h/n_ref reach 6 at c = 0.12 against at most
3 in training, where the SiLU readout is in its linear regime.

### The Gamma plateau: MD side checked, packing term tried (2026-09-25)

**MD side.** `python -m learn.gamma_md` recomputes S_NN(k) frame by frame on
the full zero-field trajectories (the c = 0.06, 0.08, 0.12 runs were extended
to 40 000 / 60 000 / 60 000 tau) and takes the error of S_NN(k_1) from blocks
inflated by the autocorrelation time of |rho_N(k,t)|^2.  Gamma = 2n/S_NN(k_1):

| c | frames (tau) | tau_int of the k_1 mode | Gamma +- err | halves | first 20 000 tau | k_2 shell |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 20 000 | 670 | 1.14 +- 0.16 | 1.26 / 1.04 | 1.14 | 1.21 |
| 0.02 | 20 000 | 470 | 1.97 +- 0.26 | 2.12 / 1.83 | 1.97 | 1.74 |
| 0.04 | 20 000 | 350 | 4.05 +- 0.63 | 3.11 / 5.78 | 4.05 | 3.50 |
| 0.05 | 20 000 | 320 | 5.66 +- 0.57 | 5.83 / 5.50 | 5.66 | 4.71 |
| 0.06 | 40 000 | 500 | 5.65 +- 0.54 | 5.74 / 5.56 | 5.74 +- 0.83 | 6.11 |
| 0.08 | 60 000 | 490 | 9.74 +- 0.75 | 10.11 / 9.39 | 9.70 +- 1.37 | 8.66 |
| 0.12 | 60 000 | 1100 | 13.4 +- 1.7 | 11.7 / 15.7 | 10.2 +- 2.3 | 12.0 |

The k_1 number mode decorrelates in 300-1100 tau, not the 1/(D k^2) ~ 1e4 tau
one would guess from the ion self-diffusion: the salt density mode relaxes
collectively, much faster than a tagged ion moves.  So the runs are 20-100
correlation times long, the halves agree within errors, and the step from
5.65 +- 0.54 at c = 0.06 to 9.74 +- 0.75 at c = 0.08 is real at the 4-sigma
level.  At c = 0.12 the earlier 11.2 (from the first 20 000 tau) was low; the
full run gives 13.4 +- 1.7 with some drift between halves.  The k_2 shell
lies within 15% of k_1 everywhere, so the k -> 0 extrapolation is not what
makes the high-c values large.  The Gamma plateau is a model problem.

**Packing term (V2b).** `ModelConfig(variant="v2b")` adds
int d^3r n_w [-ln(1 - eta)], n_w = sum_a (B * n_a), eta = sum_a v_a (B * n_a),
with B a Gaussian of width 0.5 sigma, v_a = v_max sigmoid(theta_a) learnable,
eta soft-clipped below 0.95.  It is a radial kernel plus a pointwise scalar,
passes the derivative, equivariance and Stillinger-Lovett checks, and
diverges as eta -> 1, so the extrapolation in density is set by excluded
volume rather than by the network's activation.  Trained on 0.01-0.06 with
the validation split it is indistinguishable from V2: held-out chi^2 0.607
vs 0.609, c = 0.05 prediction 0.69 vs 0.70, Gamma at 0.08 / 0.12 = 5.32 /
5.57 vs 5.17 / 5.30.  The learned volumes stayed at the bare ion volumes they
were initialised with (0.037, 2.26 sigma^3 against 0.034, 2.14): within the
training range the network already fits the density dependence, so the data
give the packing volumes no gradient, and at eta = 0.19 (c = 0.12) the term
contributes about 1 to Gamma.  For the term to carry the rise to 13 the
effective volumes would have to be the solvated ones (a cation with its
monomer shell is ~4 sigma^3, not 0.03), which the c <= 0.06 data cannot
determine.  Note also that the strongest c = 0.06 Gaussian well (p13, in the
validation set) has anion chi^2 8.5 for every model even with c = 0.06 in
training, and that the smoothed local densities in the training wells reach
1.4x the bulk density of c = 0.12: the high-density regime is present in the
data as narrow inhomogeneous regions, and it is where the functional is
weakest.

**Two more variants, same training set and split** (both pass the §7 checks):

| model | held-out chi^2 | predicts c = 0.05 | Gamma at 0.06 / 0.08 / 0.12 | S_ZZ rel rms at 0.08 / 0.12 | S_NN at 0.12 |
| --- | --- | --- | --- | --- | --- |
| V2 | 0.609 | 0.63 / 0.77 | 4.95 / 5.17 / 5.30 | 0.08 / 0.28 | 0.33 |
| V2b, bare ion volumes | 0.607 | 0.63 / 0.75 | 4.97 / 5.32 / 5.57 | 0.09 / 0.25 | 0.30 |
| V2b, solvated volumes (4.2, 2.5) as start | 0.607 | 0.62 / 0.73 | 5.34 / 5.89 / 6.24 | 0.08 / 0.16 | 0.24 |
| V2 with log(h/n_ref) inputs | 0.607 | 0.68 / 0.89 | 6.75 / 6.36 / 3.13 | 0.15 / 0.58 | 6.8 |
| MD | | | 5.65 / 9.74 / 13.4 | | |

All four are identical where there are data (train chi^2 0.587 for the three
V2 variants, held-out 0.607-0.609, the same 1% at c = 0.05).  They differ
only in what they extrapolate to, and the training data do not choose
between them: the packing volumes stay where they are initialised (4.35 and
2.67 from a start at 4.2 and 2.5) because the network absorbs whatever the
packing term does inside 0.01-0.06.  With solvated volumes the term carries
Gamma to 6.2 at c = 0.12 and halves the S_ZZ error there, still far from
13.4; log inputs make the extrapolation worse, not better.  So: the
functional is determined by the data only up to c ~ 0.06, and the prior
shape of the excluded-volume term is what sets it beyond.  The c = 0.08
field runs (12 of 16 still running on 2026-09-25) are what will pin the
high-density regime; until then the model to quote is V2 (or V2b, which
costs nothing and has the better-motivated asymptotics), with its
predictions restricted to c <= 0.06.

### Test-set figures (`learn/figures.py`)

`python -m learn.figures` writes `runs/learn/figures/`: `test_profiles.png`
(the most demanding test run per concentration, MD with its 2-sigma band
against PB, V1, V2, V3), `test_errors.png` (profile error and force residual
of every one of the 32 test runs, all models), `test_structure.png` (S_ZZ
and S_NN at all seven concentrations, three models and PB), and
`test_metrics.json`.  Models: the four-concentration fits, validation-stopped
for the networks.  Mean profile error / force chi^2 per concentration:

| model | 0.01 | 0.02 | 0.04 | 0.05 (never trained) | 0.06 |
| --- | --- | --- | --- | --- | --- |
| PB | 0.12 / 3.9 | 0.19 / 40 | 0.30 / 233 | 0.29 / 733 | 0.21 / 389 |
| V1 | 0.065 / 1.5 | 0.051 / 2.2 | 0.017 / 0.83 | 0.011 / 0.78 | 0.010 / 0.99 |
| V2 | 0.031 / 0.56 | 0.017 / 0.61 | 0.011 / 0.64 | 0.011 / 0.70 | 0.008 / 0.63 |
| V3 | 0.039 / 0.57 | 0.020 / 0.63 | 0.012 / 0.65 | 0.013 / 0.84 | 0.007 / 0.64 |

### Five concentrations and the V1 -> V2 case (`learn/present.py`, 2026-09-25)

The c = 0.08 field runs finished and passed (16/16), so the training set is
now c = 0.01, 0.02, 0.04, 0.06, 0.08 (62 PASS runs; 20 held-out; c = 0.05
never trained).  `python -m learn.present` draws
`runs/learn/figures/v2_vs_v1.png`, the one figure that makes the case for
the nonlinearity.  Numbers behind it:

| | 0.01 | 0.02 | 0.04 | 0.05 (never trained) | 0.06 | 0.08 |
| --- | --- | --- | --- | --- | --- | --- |
| test chi^2, V1 fitted on all five | 1.77 | 2.91 | 1.58 | 1.72 | 0.70 | 2.48 |
| test chi^2, V2 fitted on all five | 0.57 | 0.62 | 0.61 | 0.66 | 0.64 | 0.88 |
| leave-one-out chi^2, V1 (trained on the other four) | 1.76 | 3.33 | 1.69 | | 3.17 | 14.3 |
| leave-one-out chi^2, V2 | 0.61 | 0.64 | 0.86 | | 0.94 | 16.9 |
| leave-one-out profile error, V1 / V2 | 7.7% / 3.9% | 5.6% / 2.2% | 2.8% / 1.5% | | 1.5% / 1.5% | 4.1% / 2.9% |

A density-independent kernel fitted to five concentrations is above the
noise at every one of them (its held-out residual went 1.14 -> 1.39 -> 1.89
as concentrations were added); V2 stays at 0.57-0.66 and reaches 0.88 at
c = 0.08.  Left out and predicted from the other four, V2 is at the noise
for every concentration inside the range, including the extrapolation down
to c = 0.01 where V1's anion profiles are 9% off with 31% errors at the
minima.  Extrapolating upward to c = 0.08 fails for both (chi^2 14-17): the
0.08 state is not reachable from 0.01-0.06 by either form.

**The long-wavelength number fluctuations, compared properly.**  The MD
Gamma is 2n/S_NN(k_1) at the box wavevector k_1 = 2 pi/L ~ 0.25, and the
field data drive k >= 0.75 only, so the models must be compared at k_1, not
at their k = 0 limit (for both models the two coincide within 3%).  At k_1:

| c | 0.01 | 0.02 | 0.04 | 0.05 | 0.06 | 0.08 | 0.12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MD, 2n/S_NN(k_1) | 1.14 +- 0.16 | 1.97 +- 0.26 | 4.05 +- 0.63 | 5.66 +- 0.57 | 5.65 +- 0.54 | 9.74 +- 0.75 | 13.4 +- 1.7 |
| V1, five-concentration fit | 2.13 | 3.26 | 5.45 | 6.50 | 7.52 | 9.43 | 12.8 |
| V2, five-concentration fit | 1.10 | 1.74 | 3.35 | 3.95 | 4.45 | 5.40 | 5.31 |

V2 is right at c <= 0.04 and low by 40-45% at 0.08 and 0.12, even with
c = 0.08 in its training set and its force residual there at 0.88; V1 is
right at c >= 0.08 and high by 90% at 0.01.  Neither is constrained at k_1
by the field profiles (no driven mode below k = 0.75), so what each does at
k_1 is its own extrapolation from k >= 0.75: V1's is Gamma - 1 proportional
to n, which happens to be the truth above c = 0.06; V2's density dependence,
learned from the anion channel at k >= 0.75, flattens at k_1.  The zero-field
S(k) at the box wavevectors is therefore independent information that the
training does not use; adding S_NN(k_1), S_ZZ(k_1) at the seven
concentrations to the loss (they now have error bars) is the direct fix,
and the one to try next.

**Why V2 is wrong at k_1 and V1 is not** (`learn/diagnose_k.py`,
`runs/learn/figures/k_diagnostic.png`).  The training data are only the
field-run profiles: n_a(z) and f^int_a(z) on the 0.1-sigma grid for the
62 PASS runs, nothing else.  Their potentials drive m in {3,4,5,6} (Fourier),
plus {8,...,20} (mixed), plus the harmonics of 4 (Gaussian trains): every
driven mode has k >= 0.77.  The m = 1, 2 components of the profiles are the
slow undriven thermal modes, 20-100x weaker than the driven ones.  Plotting
2n/S_NN(k) - 1 from MD against the same quantity from each model's own
S_NN(k) shows that V2 reproduces MD at every k >= 0.77 at every trained
concentration (at c = 0.08: 4.28 / 3.24 / 1.96 vs MD 4.74 / 3.29 / 1.81 at
k = 0.75 / 1.0 / 1.3) and goes flat below k = 0.75, where MD keeps rising to
8.9 at k_1.  V1 follows that rise (8.4 at k_1) but is worse inside the
driven band (2.89 vs 3.29 at k = 1.0, 0.65 vs 0.84 at k = 2.0) and is off by
a factor 8 at k_1 for c = 0.01.  So V2 is exactly as good as the data it was
given, and the missing piece is a long-wavelength (range > 8 sigma) part of
the number-channel kernel at c >= 0.04 that no field run probes.  V1 gets it
by the shape of a Gaussian sum shared across concentrations, not from
information.  Two remedies, both cheap: put S_NN(k_1), S_ZZ(k_1) at the seven
concentrations into the loss, and/or add field runs whose potentials drive
m = 1 and 2.

### Long-wavelength field runs (submitted 2026-09-25, jobs 17754916-17754950)

The learned functional is unconstrained below k = 0.75 because no production
potential drives m = 1, 2 (the schedule avoided m = 1 on purpose, for its
tracer-estimated transient).  The zero-field runs now show the k_1 salt-density
mode decorrelating in 300-1100 tau, so a supplement was added instead of
putting S(k_1) into the loss: `make_potential.py --family long` (both of
m = 1, 2, wavelengths L and L/2) and `long3` (m = 1, 2, 3),
`make_schedule.py --mode long` emitting six runs per state point: two V_N
only, two both parts, one both with m = 3, one psi only; peak-to-peak
0.7-3 kT.  Same seeds at every concentration (770016-770021), tags p16-p21
(p24-p29 at the c = 0.04 pilot), c = 0.02, 0.04, 0.05, 0.06, 0.08, run lengths
as the production runs of each state point, 64 ranks on scavenger with
checkpoint/requeue.  The c = 0.05 ones are test data only.

The c = 0.05, 0.06, 0.08 arrays were resubmitted a few minutes later on
whole nodes (128 ranks, jobs 17755435, 17755432, 17755431): on an idle amd20
node 128 ranks give 815 steps/s against 667 for 64, and a 64-rank job that
shares its node loses another 45%, so the whole-node runs should finish in
about half the time (c = 0.08: ~12-15 h instead of ~30 h).  c = 0.02 and
0.04 stay at 64 ranks (jobs 17754949, 17754950), they are short.

The follow-up is queued as SLURM dependencies: analysis arrays
17755436-17755438 and 17755006-17755007 (`analyze_field_msu.sbatch`,
afterok the field arrays) and then `slurm/learn_long.sbatch` (job 17755439),
which refreshes the
acceptance table, fits V1 / V2 / V3 on the five concentrations with the new
runs (experiment names carry the suffix `_long`), predicts c = 0.05, and
writes `k_diagnostic_long.png` and `v2_vs_v1_long.png`.  What to look at
when it has run: the bottom row of `k_diagnostic_long.png` for k < 0.75 at
c = 0.04-0.08, and Gamma at k_1 in `v2_vs_v1_long.json`.  If a field task
fails, the dependent jobs stay pending; run the three steps by hand.

### Result of the long-wavelength runs (2026-09-25 evening)

All 30 runs finished (the c = 0.08 ones in 10-15.5 h on whole nodes) and
passed acceptance (30/30).  Training set: the five concentrations with the
long runs included, 92 PASS runs, c = 0.05 never trained.  Two rounds:
plain force matching (`_long`) and the same with the m = 1, 2 modes of the
long-family runs weighted by (k_3/k)^2 in the loss (`--kboost 2`, `_longk`;
the same identity and data, different mode weights; V1's least squares does
not use it).

| model | held-out chi^2 (20 runs) | rel L2 | c = 0.05, all 22 runs incl. 6 long: chi^2, rel L2 |
| --- | --- | --- | --- |
| V2 before the long runs | 0.664 | 0.018 / 0.015 | 0.661, 0.012 / 0.009 |
| V2, plain loss | 0.654 | 0.017 / 0.014 | 0.639, 0.010 / 0.009 |
| V2, weighted loss | 0.658 | 0.016 / 0.014 | 0.634, 0.010 / 0.008 |
| V3, plain loss | 0.882 | 0.020 / 0.016 | 0.827 |
| V1 | 1.887 | 0.037 / 0.048 | 1.416 |

The long-wavelength kernel, 2n/S_NN(k_1) - 1 at the box wavevector:

| c | MD | V2 before | V2 plain | V2 weighted |
| --- | --- | --- | --- | --- |
| 0.01 | 0.14 +- 0.17 | 0.10 | 0.39 | 0.36 |
| 0.02 | 0.97 +- 0.27 | 0.74 | 1.07 | 1.06 |
| 0.04 | 3.05 +- 0.65 | 2.35 | 3.05 | 3.25 |
| 0.05 (never trained) | 4.66 +- 0.58 | 2.95 | 4.04 | 4.36 |
| 0.06 | 4.65 +- 0.55 | 3.45 | 5.08 | 5.54 |
| 0.08 | 8.74 +- 0.76 | 4.40 | 7.27 | 7.85 |
| 0.12 (no data) | 12.4 +- 1.7 | 4.31 | 6.68 | 7.44 |

The plain loss already does it: inside the training range every value is
within 1-2 sigma of MD, c = 0.08 goes from a factor 2 low to 1.2-1.9 sigma
low, and the m = 1 response amplitude on the long runs goes from 20-45% off to
5-10% (c = 0.05 included).  The worry that force matching cannot see k_1
was overstated: one bin's residual there is small, but the m = 1 component
of 30 runs is coherent and the gradient follows it.  The weighting adds
0.1-0.6 at k_1 and nothing elsewhere, so the plain loss is the one to keep
(`k_diagnostic_long.png`, `v2_vs_v1_long.png`).  c = 0.12 remains an
extrapolation, half repaired.

Two costs, in regions without training data: the strongest c = 0.08 well
(p01, contrast ~100, in the validation set) went from chi^2 9.5 to 38 for
both species, and the S_ZZ extrapolation to c = 0.12 broke (rel rms 0.17 ->
1.3, stability margin 0.16 at k = 1.8).  The network's capacity (L = 1,
C = 4, readout 64x64) is now the limit: the long-wavelength behaviour was
bought at the expense of the highest local densities.  Next: L = 2 or a
wider readout on this data set, and c = 0.12 field runs if the extrapolation
there matters.

Leave-one-concentration-out repeated with the long runs (the left-out
concentration now includes its six long runs; `transfer_*_val3_long`,
`v2_vs_v1_long.png`):

| left out | 0.01 | 0.02 | 0.04 | 0.06 | 0.08 |
| --- | --- | --- | --- | --- | --- |
| V1: chi^2 / profile rel L2 | 1.99 / 7.9% | 3.07 / 5.7% | 1.57 / 2.6% | 1.23 / 1.2% | 6.2 / 3.0% |
| V2: chi^2 / profile rel L2 | 0.66 / 3.9% | 0.61 / 2.1% | 1.07 / 2.0% | 1.82 / 1.4% | 9.2 / 2.1% |

V2 predicts a left-out concentration from its neighbours at or near the
noise up to c = 0.06 and is 2-5x better than V1 in the profiles at the low
end; the upward extrapolation to c = 0.08 still fails for both (V2 worse in
chi^2, better in the profiles).  With the long runs in the target set the
0.06 case is harder than before (0.94 -> 1.82): the long-wavelength kernel
has to be extrapolated in density as well.

**Scope (decided 2026-09-25).** The functional is to be used and judged
inside the data: c = 0.01-0.08 and k >= k_1 = 2 pi / L.  No extrapolation to
c = 0.12 is claimed or pursued, and the c = 0.12 columns in the tables above
are reported only as what the model does there, not as a target.  Inside the
range the model `joint_v2_L1_C4_c001_c002_c004_c006_c008_val3_long` agrees
with MD at every tested concentration and wavevector; the one in-range gap
is the strongest c = 0.08 well (p01, contrast ~100, chi^2 38 as a validation
run), i.e. the highest local densities.

### Capacity on the full data set (2026-09-25, late)

Same data (five concentrations + long runs), same validation split, plain
loss.  `--readout-hidden` and `--hidden` set the widths (names carry
`_R<widths>_H<hidden>` when non-default).

| model | train | held-out chi^2 | held-out at 0.08 | c = 0.05 (22 runs) | validation p01@0.08 (strongest well) | p07@0.08 | 2n/S_NN(k_1)-1 at 0.08 (MD 8.74) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L = 1, readout 64x64 | 0.616 | 0.654 | 0.91 | 0.639 | 38.6 / 37.4 | 9.2 / 10.0 | 7.27 |
| L = 2, C = 4 | 0.617 | 0.727 | 1.22 | 0.653 | 102 / 99 | 26.8 / 19.7 | 7.40 |
| L = 1, readout 128x128 | 0.607 | 0.629 | 0.79 | 0.623 | 22.3 / 24.1 | 4.7 / 5.1 | 7.47 |

Width helps, depth hurts: the wider readout halves the residual of the
strongest c = 0.08 wells and improves everything else slightly, with the
long-wavelength kernel unchanged; a second convolution layer makes the
high-density regime worse.  The limit is the pointwise nonlinearity Phi, not
the receptive field.  Current best: `joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3_long`.
The strongest well is still at chi^2 22, so a further step in width
(256x256 or three layers of 64) is the natural next experiment.
