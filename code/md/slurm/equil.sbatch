#!/bin/bash
# Stages 1-3 for one state point: build -> soft push-off -> warm-up -> NPT(p=0).
#SBATCH -J sip_eq
#SBATCH -A mth210003p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 12:00:00
set -e
source /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/env.sh
source /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/slurm/_check_ranks.sh
: ${RUNDIR:?}; : ${CONC:?}; : ${TEMP:=1.0}; : ${SEED:=20260920}
: ${NPUSH:=60000}; : ${NCAP:=20000}; : ${NNVT:=400000}; : ${NNPT:=2000000}
: ${PDAMP:=1.0}; : ${RHO0:=0.85}; : ${PPPM:=1.0e-4}
NP=${SLURM_NTASKS:-32}
FF=$SIP_ROOT/templates/ff.lmp
TAB=$SIP_ROOT/tools/solvation.table
mkdir -p $RUNDIR && cd $RUNDIR
echo "### state point: c=$CONC T=$TEMP seed=$SEED  on $NP ranks"

$SIP_PY $SIP_ROOT/tools/build_system.py --conc $CONC --rho $RHO0 --seed $SEED --out melt.data

mpirun -n $NP $LMP -log log.01_pushoff -in $SIP_ROOT/templates/in.01_pushoff \
       -var data melt.data -var out restart.pushoff -var nsteps $NPUSH \
       -var T $TEMP -var seed $SEED
check_ranks log.01_pushoff $NP

mpirun -n $NP $LMP -log log.02_warmup -in $SIP_ROOT/templates/in.02_warmup \
       -var restart restart.pushoff -var ff $FF -var tabfile $TAB \
       -var out restart.warmup -var T $TEMP -var seed $((SEED+1)) \
       -var ncap $NCAP -var nnvt $NNVT -var pppm_acc $PPPM
check_ranks log.02_warmup $NP

mpirun -n $NP $LMP -log log.03_npt -in $SIP_ROOT/templates/in.03_npt \
       -var restart restart.warmup -var ff $FF -var tabfile $TAB \
       -var out restart.npt -var T $TEMP -var pdamp $PDAMP -var nsteps $NNPT -var pppm_acc $PPPM
check_ranks log.03_npt $NP

echo "### equil done"; grep FINAL_L log.03_npt || true
