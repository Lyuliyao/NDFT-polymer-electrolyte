#!/bin/bash
# Stage 4: zero-field NVT production at the mean NPT volume.
#SBATCH -J sip_zf
#SBATCH -A mth210003p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 24:00:00
set -e
source /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/env.sh
source /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/slurm/_check_ranks.sh
: ${RUNDIR:?}; : ${LT:?}; : ${SRCRESTART:?}; : ${TEMP:=1.0}
: ${NEQUIL:=200000}; : ${NSTEPS:=4000000}; : ${NDUMP:=200}; : ${PPPM:=1.0e-4}
NP=${SLURM_NTASKS:-32}
mkdir -p $RUNDIR && cd $RUNDIR

mpirun -n $NP $LMP -log log.04_zerofield -in $SIP_ROOT/templates/in.04_zerofield \
       -var restart $SRCRESTART -var ff $SIP_ROOT/templates/ff.lmp \
       -var tabfile $SIP_ROOT/tools/solvation.table -var out restart.zf \
       -var T $TEMP -var Lt $LT -var nequil $NEQUIL -var nsteps $NSTEPS \
       -var ndump $NDUMP -var pppm_acc $PPPM
check_ranks log.04_zerofield $NP
echo "### zerofield done (dump left uncompressed for analysis)"
