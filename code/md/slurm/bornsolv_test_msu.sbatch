#!/bin/bash --login
# MSU HPCC (amd20) port of slurm/bornsolv_test.sbatch.
# Test the solvation model with Hall's own pair style (bornsolv), against the
# paper's Fig. S1a (V), Fig. S2 (D+, D-) and Fig. S3 (g(r)).  c_LJ = 0.04, T* = 1.
#   array 0  one_cation : sigma = 1, cation-monomer only   (leading hypothesis)
#   array 1  ij_both    : sigma_ij, both ions               (eq. 3 as written; control)
#   array 2  one_both   : sigma = 1, both ions              (Hall bornsolv convention)
#   array 3  ij_cation  : sigma_ij, cation-monomer only
# Run a subset with e.g. `sbatch --array=2-3 slurm/bornsolv_test_msu.sbatch`.
# All start from the converged hanhai20 NPT state of the current model.
#SBATCH --job-name=sip_bstest
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/bornsolv_test/%x-%A_%a.SLURMout
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU   # stale values from a submitting
# environment that is itself inside a SLURM job collide with --mem-per-cpu in srun
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
LMP=$LMP_BS
NP=${SLURM_NTASKS:-32}
VARIANTS=(one_cation ij_both one_both ij_cation)
V=${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}
FF=$SIP_ROOT/templates/ff_bs_$V.lmp
RUN=$SIP_ROOT/runs/bornsolv_test/$V
mkdir -p $RUN && cd $RUN
echo "### variant $V on $NP ranks, $(hostname), $(date)"
echo "### binary $LMP"

run() { local log=$1; shift; srun -n $NP $LMP -log $log "$@" > /dev/null; check_ranks $log $NP; }

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
srun -n 1 $LMP -log log.snap -in $SIP_ROOT/templates/in.restart2data \
     -var restart restart.zf -var ff $FF -var out final.data > /dev/null
$SIP_PY $SIP_ROOT/tools/analyze_zerofield.py --npt . --run . --conc 0.04 \
        --out result.json --sk-frames 100 | tee analysis.txt
$SIP_PY $SIP_ROOT/tools/rdf_snapshot.py final.data --out rdf.dat | tee -a analysis.txt
echo "### variant $V done $(date)"

set +e  # accounting only; never fail the job here (powertools has no `js` on this cluster)
scontrol show job $SLURM_JOB_ID
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State
exit 0
