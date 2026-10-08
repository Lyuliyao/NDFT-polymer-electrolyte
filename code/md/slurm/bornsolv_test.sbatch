#!/bin/bash
# Test the solvation model with Hall's own pair style (bornsolv), against the
# paper's Fig. S1a (V), Fig. S2 (D+, D-) and Fig. S3 (g(r)).  c_LJ = 0.04, T* = 1.
#   array 0  one_cation : sigma = 1, cation-monomer only   (leading hypothesis)
#   array 1  ij_both    : sigma_ij, both ions               (eq. 3 as written; control)
# Both start from the converged hanhai20 NPT state of the current model.
#SBATCH -J sip_bstest
#SBATCH -A mth210003p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 02:00:00
#SBATCH --array=0-1
#SBATCH -o /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/runs/bornsolv_test/job.%A_%a.out
set -e
source /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/env.sh
source $SIP_ROOT/slurm/_check_ranks.sh
LMP=$SIP_ROOT/build/lammps_bornsolv/build-bs/lmp_bornsolv
NP=${SLURM_NTASKS:-32}
VARIANTS=(one_cation ij_both)
V=${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}
FF=$SIP_ROOT/templates/ff_bs_$V.lmp
RUN=$SIP_ROOT/runs/bornsolv_test/$V
mkdir -p $RUN && cd $RUN
echo "### variant $V on $NP ranks, $(hostname), $(date)"

# --- find a launcher that forms ONE communicator (srun gives singletons here) ---
LAUNCH=""
for L in "mpirun -n $NP" "srun --mpi=pmi2 -n $NP"; do
  rm -f log.probe
  $L $LMP -log log.probe -in $SIP_ROOT/templates/in.mpiprobe > /dev/null 2>&1 || true
  got=$(grep -m1 -oE "[0-9]+ MPI tasks" log.probe 2>/dev/null | awk '{print $1}')
  echo "  launcher '$L' -> '${got:-none}' ranks"
  [ "$got" = "$NP" ] && { LAUNCH="$L"; break; }
done
[ -n "$LAUNCH" ] || { echo "FATAL: no launcher gave $NP ranks"; exit 1; }
echo "### using: $LAUNCH"

run() { local log=$1; shift; $LAUNCH $LMP -log $log "$@"; check_ranks $log $NP; }

# 0. import the data file into this binary's restart format
run log.00_import -in $SIP_ROOT/templates/in.data2restart \
    -var data $SIP_ROOT/runs/bornsolv_test/equil_hanhai20.data -var ff $FF -var out restart.import

# 1. switch potential safely: capped Langevin 40 tau, then Nose-Hoover NVT 500 tau
run log.02_warmup -in $SIP_ROOT/templates/in.02_warmup \
    -var restart restart.import -var ff $FF -var out restart.warm \
    -var seed $((8100 + ${SLURM_ARRAY_TASK_ID:-0})) -var ncap 20000 -var nnvt 100000

# 2. NPT at p = 0, 500 tau (the volume converged within ~200 tau previously)
run log.03_npt -in $SIP_ROOT/templates/in.03_npt \
    -var restart restart.warm -var ff $FF -var out restart.npt -var nsteps 100000
LT=$($SIP_PY -c "import numpy as np; d=np.loadtxt('vol.dat',comments='#'); print(f'{d[len(d)//3:,1].mean():.6f}')")
echo "### $V  <L> = $LT"

# 3. zero-field NVT at <L>: 2000 tau of ion trajectories -> D+, D-, sigma/sigma_NE
run log.04_zerofield -in $SIP_ROOT/templates/in.04_zerofield \
    -var restart restart.npt -var ff $FF -var out restart.zf \
    -var Lt $LT -var nequil 20000 -var nsteps 400000

# 4. analysis
$LMP -log log.snap -in $SIP_ROOT/templates/in.restart2data \
     -var restart restart.zf -var ff $FF -var out final.data > /dev/null
$SIP_PY $SIP_ROOT/tools/analyze_zerofield.py --npt . --run . --conc 0.04 \
        --out result.json --sk-frames 100 | tee analysis.txt
$SIP_PY $SIP_ROOT/tools/rdf_snapshot.py final.data --out rdf.dat | tee -a analysis.txt
echo "### variant $V done $(date)"
