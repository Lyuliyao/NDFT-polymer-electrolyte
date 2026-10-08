#!/bin/bash --login
# Stage 5: one external-potential production run (submit as a job array).
# MSU HPCC (amd20) counterpart of field.sbatch; run_state.sh submits it when SIP_SITE=msu.
#SBATCH --job-name=sip_fld
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --mail-user=lyuliyao@msu.edu
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU   # stale values from a submitting
# environment that is itself inside a SLURM job collide with --mem-per-cpu in srun
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${STATEDIR:?}; : ${TEMP:=1.0}
: ${SRCRESTART:=$STATEDIR/zerofield/restart.zf}
: ${NTRANS:=1100000}; : ${NSTEPS:=11000000}; : ${NDUMP:=200}
: ${BINW:=0.1}; : ${NREP:=500}; : ${PPPM:=1.0e-4}
# force field and solvation table; a state point with its own (e.g. eps_r = 2) passes them
: ${FF:=$SIP_ROOT/templates/ff.lmp}; : ${TAB:=$SIP_ROOT/tools/solvation.table}
NP=${SLURM_NTASKS:-32}
: ${NRST:=500000}
# in.05_field_ck also checkpoints the transient (trans.<step>); the default keeps the
# template every earlier run used
: ${TEMPLATE:=in.05_field}
TAG=$(printf "p%02d" ${SLURM_ARRAY_TASK_ID:-0})
RUNDIR=$STATEDIR/field_$TAG
mkdir -p $RUNDIR && cd $RUNDIR
test -f $SRCRESTART || { echo "missing restart $SRCRESTART"; exit 1; }
test -f $STATEDIR/potentials/$TAG.lmp || { echo "missing potential $TAG"; exit 1; }

# --- resume after a preemption (scavenger) or a time-limit kill --------------
# `restart N restart.*` names each checkpoint by its timestep, so the newest one
# is simply the largest number, with no ambiguity about which of two files is
# complete.  The dumps are cut back to it, because they hold frames written
# after the checkpoint that the resumed run will write again.
# A checkpoint whose write was cut off (seen 2026-09-27: a filesystem I/O error
# left 16-byte files) has a different size from the others; delete it first.
for pat in restart trans; do
  if ls $pat.[0-9]* >/dev/null 2>&1; then
    SZ=$(stat -c %s $pat.[0-9]* | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
    for f in $pat.[0-9]*; do
      [ "$(stat -c %s $f)" = "$SZ" ] || { echo "### dropping truncated checkpoint $f"; rm -f $f; }
    done
  fi
done
TLAST=$(ls -1 trans.[0-9]* 2>/dev/null | sed 's/.*trans\.//' | sort -n | tail -1)
LAST=$(ls -1 restart.[0-9]* 2>/dev/null | sed 's/.*restart\.//' | sort -n | tail -1)
if [ -n "$LAST" ]; then
  RESUME=1
  START=restart.$LAST
  SEG=$(ls prof_cat.*.dat 2>/dev/null | wc -l)
  echo "### resuming $TAG from step $LAST (segment $SEG)"
  ls restart.[0-9]* | grep -v "^restart\.$LAST$" | xargs -r rm -f
  rm -f trans.[0-9]*                       # the transient is over
  for f in cat.dump ani.dump; do $SIP_PY $SIP_ROOT/tools/dump_truncate.py $f $LAST; done
  LOGF=log.05_field.$SEG
elif [ "$TEMPLATE" = in.05_field_ck ] && [ -n "$TLAST" ]; then
  RESUME=2
  START=trans.$TLAST
  SEG=0
  ls trans.[0-9]* | grep -v "^trans\.$TLAST$" | xargs -r rm -f
  LOGF=log.05_field.0.transient_resume$(ls log.05_field.0* 2>/dev/null | wc -l)
  echo "### resuming the transient of $TAG from step $TLAST of $NTRANS"
else
  RESUME=0
  START=$SRCRESTART
  SEG=0
  LOGF=log.05_field.0
  echo "### fresh start for $TAG from $SRCRESTART"
fi

srun -n $NP $LMP -log $LOGF -in $SIP_ROOT/templates/$TEMPLATE \
       -var restart $START -var resume $RESUME -var seg $SEG -var nrst $NRST \
       -var ff $FF -var tabfile $TAB \
       -var potfile $STATEDIR/potentials/$TAG.lmp -var out restart.field \
       -var T $TEMP -var ntrans $NTRANS -var nsteps $NSTEPS -var ndump $NDUMP \
       -var binw $BINW -var nrep $NREP -var pppm_acc $PPPM
check_ranks $LOGF $NP
# the run finished: its own restart file is the one to keep
test -f restart.field && rm -f restart.[0-9]* trans.[0-9]*
echo "### field $TAG done"
