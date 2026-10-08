#!/bin/bash --login
# Zero-field replicas on the preemptible scavenger partition (2026-10-07): as zerofield_rep_msu.sbatch, but the production
# writes checkpoints rep.<step> every NRST steps and a requeued job resumes from the newest one (ions.dump cut back to it,
# tools/dump_truncate.py), so a preemption costs at most NRST steps (or the equilibration, which is not checkpointed).
#   STATE=<state dir> ZF=<its zero-field dir> SEEDBASE=5000 sbatch --array=1-8 -o $STATE/zfr.%A_%a.out slurm/zerofield_rep_ck_msu.sbatch
#SBATCH --job-name=sip_zfr
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --constraint=amd20|amd22|amd24
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --requeue
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${STATE:?}; : ${ZF:?}; : ${SEEDBASE:=100}; : ${TEMP:=1.0}; : ${NEQUIL:=1000000}; : ${NSTEPS:=8000000}; : ${NDUMP:=1000}; : ${NRST:=500000}; : ${PPPM:=1.0e-4}
ID=${SLURM_ARRAY_TASK_ID:?}; RUNDIR=$STATE/zerofield_rep/r$ID; SEED=$((SEEDBASE + ID))
LT=$($SIP_PY -c "import json; print(f\"{json.load(open('$ZF/D.json'))['lz']:.6f}\")")
NP=${SLURM_NTASKS:-32}
mkdir -p $RUNDIR && cd $RUNDIR
if [ -f modes.npz ]; then echo "replica $ID already done"; exit 0; fi
if [ ! -f rep.final ]; then
  # drop a checkpoint whose write was cut off (size differs from the others), then pick the newest
  if ls rep.[0-9]* >/dev/null 2>&1; then
    SZ=$(stat -c %s rep.[0-9]* | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
    for f in rep.[0-9]*; do [ "$(stat -c %s $f)" = "$SZ" ] || { echo "### dropping truncated checkpoint $f"; rm -f $f; }; done
  fi
  LAST=$(ls -1 rep.[0-9]* 2>/dev/null | sed 's/.*rep\.//' | sort -n | tail -1)
  if [ -n "$LAST" ]; then
    RESUME=1; START=rep.$LAST; LOGF=log.zf_rep.resume$(ls log.zf_rep* 2>/dev/null | wc -l)
    ls rep.[0-9]* | grep -v "^rep\.$LAST$" | xargs -r rm -f
    $SIP_PY $SIP_ROOT/tools/dump_truncate.py ions.dump $LAST
    echo "### $(date): replica $ID of $STATE resumes from step $LAST"
  else
    RESUME=0; START=$STATE/equil/restart.npt; LOGF=log.zf_rep; rm -f ions.dump
    echo "### $(date): replica $ID of $STATE, seed $SEED, L = $LT, fresh start, $NEQUIL + $NSTEPS steps on $NP ranks"
  fi
  srun -n $NP $LMP -log $LOGF -in $SIP_ROOT/templates/in.04_zerofield_rep_ck \
         -var restart $START -var resume $RESUME -var nrst $NRST -var ff $SIP_ROOT/templates/ff.lmp \
         -var tabfile $SIP_ROOT/tools/solvation.table -var T $TEMP -var Lt $LT \
         -var nequil $NEQUIL -var nsteps $NSTEPS -var ndump $NDUMP -var seed $SEED -var pppm_acc $PPPM
  check_ranks $LOGF $NP
fi
$SIP_PY $SIP_ROOT/tools/zf_modes.py --run $RUNDIR && rm -f rep.[0-9]*
echo "### replica $ID done $(date)"
