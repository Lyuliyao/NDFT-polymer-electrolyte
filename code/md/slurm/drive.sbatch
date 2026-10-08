#!/bin/bash --login
# Drive pilots (step amplitude series, periodic driving), c = 0.04: one or more runs per array task.
#   sbatch --array=1-N --export=ALL,LIST=runs/drive_T1.0_c0.04/step_runs.txt,PER=2 slurm/drive.sbatch
# LIST lines: "<rundir> <seed restart> <potential .lmp> <nsteps>"; task i runs lines (i-1)*PER+1 .. i*PER.
#SBATCH --job-name=sip_drv
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${LIST:?}; : ${PER:=1}; : ${PPPM:=1.0e-4}; : ${TEMPLATE:=in.06_relax}
IN=$SIP_ROOT/runs/drive_T1.0_c0.04/inputs
NP=${SLURM_NTASKS:-32}
first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER + 1 )); last=$(( SLURM_ARRAY_TASK_ID * PER ))

one_run() {
  local RUNDIR=$1 SEED=$2 POT=$3 NSTEPS=$4
  mkdir -p $RUNDIR && cd $RUNDIR || return 1
  [ -f restart.relax ] && { echo "### $RUNDIR done"; return 0; }
  if ls restart.[0-9]* >/dev/null 2>&1; then
    local SZ=$(stat -c %s restart.[0-9]* | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
    for f in restart.[0-9]*; do [ "$(stat -c %s $f)" = "$SZ" ] || rm -f $f; done
  fi
  local LAST=$(ls -1 restart.[0-9]* 2>/dev/null | sed 's/.*restart\.//' | sort -n | tail -1)
  local RESUME START SEG
  if [ -n "$LAST" ]; then
    RESUME=1; START=restart.$LAST; SEG=$(ls ave_cat.*.dat 2>/dev/null | wc -l)
    ls restart.[0-9]* | grep -v "^restart\.$LAST$" | xargs -r rm -f
    for f in cat.dump ani.dump; do $SIP_PY $SIP_ROOT/tools/dump_truncate.py $f $LAST; done
  else
    RESUME=0; START=$SEED; SEG=0
    rm -f cat.dump ani.dump ave_*.dat snap_*.dat thermo.*.dat log.relax.*
    printf "%s\n%s\n" "$SEED" "$POT" > source.txt
  fi
  srun -n $NP $LMP -log log.relax.$SEG -in $IN/$TEMPLATE \
       -var restart $START -var resume $RESUME -var seg $SEG -var DIR z \
       -var ff $IN/ff.lmp -var tabfile $IN/solvation.table -var pppm_acc $PPPM \
       -var potfile $POT -var nsteps $NSTEPS -var out restart.relax
  check_ranks log.relax.$SEG $NP || return 1
  [ -f restart.relax ] && rm -f restart.[0-9]*
  echo "### $RUNDIR finished $(date)"
}

for i in $(seq $first $last); do
  line=$(sed -n "${i}p" $SIP_ROOT/$LIST)
  [ -z "$line" ] && continue
  set -- $line
  cd $SIP_ROOT
  R=$1; [ "${R:0:1}" = / ] || R=$SIP_ROOT/$R
  P=$3; [ "${P:0:1}" = / ] || P=$SIP_ROOT/$P
  one_run $R $2 $P $4
done
