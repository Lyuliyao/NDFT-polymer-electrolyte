#!/bin/bash
# Stage 5: one external-potential production run (submit as a job array).
#SBATCH -J sip_fld
#SBATCH -A mth210003p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 48:00:00
set -e
source /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/env.sh
source /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/slurm/_check_ranks.sh
: ${STATEDIR:?}; : ${TEMP:=1.0}
: ${SRCRESTART:=$STATEDIR/zerofield/restart.zf}
: ${NTRANS:=1100000}; : ${NSTEPS:=11000000}; : ${NDUMP:=200}
: ${BINW:=0.1}; : ${NREP:=500}; : ${PPPM:=1.0e-4}
NP=${SLURM_NTASKS:-32}
TAG=$(printf "p%02d" ${SLURM_ARRAY_TASK_ID:-0})
RUNDIR=$STATEDIR/field_$TAG
mkdir -p $RUNDIR && cd $RUNDIR
test -f $SRCRESTART || { echo "missing restart $SRCRESTART"; exit 1; }
test -f $STATEDIR/potentials/$TAG.lmp || { echo "missing potential $TAG"; exit 1; }

mpirun -n $NP $LMP -log log.05_field -in $SIP_ROOT/templates/in.05_field \
       -var restart $SRCRESTART -var ff $SIP_ROOT/templates/ff.lmp \
       -var tabfile $SIP_ROOT/tools/solvation.table \
       -var potfile $STATEDIR/potentials/$TAG.lmp -var out restart.field \
       -var T $TEMP -var ntrans $NTRANS -var nsteps $NSTEPS -var ndump $NDUMP \
       -var binw $BINW -var nrep $NREP -var pppm_acc $PPPM
check_ranks log.05_field $NP
echo "### field $TAG done"
