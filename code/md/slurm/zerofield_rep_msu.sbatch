#!/bin/bash --login
# Zero-field replicas (2026-10-06): independent NVT runs per state point from the NPT restart with fresh velocities,
# 5000 tau equilibration + 40000 tau production, ion positions every NDUMP steps, then the Fourier modes of the ion
# densities (tools/zf_modes.py -> modes.npz).  Pooled over replicas by tools/zf_pool.py (Gamma(k_1), S(k) at small k).
#   STATE=<state dir> ZF=<its zero-field dir> SEEDBASE=100 sbatch --array=1-8 -o $STATE/zfr.%A_%a.out slurm/zerofield_rep_msu.sbatch
#SBATCH --job-name=sip_zfr
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20|amd22|amd24
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${STATE:?}; : ${ZF:?}; : ${SEEDBASE:=100}; : ${TEMP:=1.0}; : ${NEQUIL:=1000000}; : ${NSTEPS:=8000000}; : ${NDUMP:=1000}; : ${PPPM:=1.0e-4}
ID=${SLURM_ARRAY_TASK_ID:?}; RUNDIR=$STATE/zerofield_rep/r$ID; SEED=$((SEEDBASE + ID))
LT=$($SIP_PY -c "import json; print(f\"{json.load(open('$ZF/D.json'))['lz']:.6f}\")")
NP=${SLURM_NTASKS:-32}
mkdir -p $RUNDIR && cd $RUNDIR
if [ -f modes.npz ]; then echo "replica $ID already done"; exit 0; fi
echo "### $(date): replica $ID of $STATE, seed $SEED, L = $LT, $NEQUIL + $NSTEPS steps on $NP ranks"
srun -n $NP $LMP -log log.zf_rep -in $SIP_ROOT/templates/in.04_zerofield_rep \
       -var restart $STATE/equil/restart.npt -var ff $SIP_ROOT/templates/ff.lmp \
       -var tabfile $SIP_ROOT/tools/solvation.table -var T $TEMP -var Lt $LT \
       -var nequil $NEQUIL -var nsteps $NSTEPS -var ndump $NDUMP -var seed $SEED -var pppm_acc $PPPM
check_ranks log.zf_rep $NP
$SIP_PY $SIP_ROOT/tools/zf_modes.py --run $RUNDIR
echo "### replica $ID done $(date)"
