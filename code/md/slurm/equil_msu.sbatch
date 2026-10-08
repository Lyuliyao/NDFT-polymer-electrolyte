#!/bin/bash --login
# Stages 1-3 for one state point: build -> soft push-off -> warm-up -> NPT(p=0).
# MSU HPCC (amd20) counterpart of equil.sbatch; run_state.sh submits it when SIP_SITE=msu.
#SBATCH --job-name=sip_eq
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lyuliyao@msu.edu
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU   # stale values from a submitting
# environment that is itself inside a SLURM job collide with --mem-per-cpu in srun
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${RUNDIR:?}; : ${CONC:?}; : ${TEMP:=1.0}; : ${SEED:=20260920}
: ${NPUSH:=60000}; : ${NCAP:=20000}; : ${NNVT:=400000}; : ${NNPT:=2000000}
: ${PDAMP:=1.0}; : ${RHO0:=0.85}; : ${PPPM:=1.0e-4}
NP=${SLURM_NTASKS:-32}
FF=$SIP_ROOT/templates/ff.lmp
TAB=$SIP_ROOT/tools/solvation.table
mkdir -p $RUNDIR && cd $RUNDIR
echo "### state point: c=$CONC T=$TEMP seed=$SEED  on $NP ranks"

$SIP_PY $SIP_ROOT/tools/build_system.py --conc $CONC --rho $RHO0 --seed $SEED --out melt.data

srun -n $NP $LMP -log log.01_pushoff -in $SIP_ROOT/templates/in.01_pushoff \
       -var data melt.data -var out restart.pushoff -var nsteps $NPUSH \
       -var T $TEMP -var seed $SEED
check_ranks log.01_pushoff $NP

srun -n $NP $LMP -log log.02_warmup -in $SIP_ROOT/templates/in.02_warmup \
       -var restart restart.pushoff -var ff $FF -var tabfile $TAB \
       -var out restart.warmup -var T $TEMP -var seed $((SEED+1)) \
       -var ncap $NCAP -var nnvt $NNVT -var pppm_acc $PPPM
check_ranks log.02_warmup $NP

srun -n $NP $LMP -log log.03_npt -in $SIP_ROOT/templates/in.03_npt \
       -var restart restart.warmup -var ff $FF -var tabfile $TAB \
       -var out restart.npt -var T $TEMP -var pdamp $PDAMP -var nsteps $NNPT -var pppm_acc $PPPM
check_ranks log.03_npt $NP

echo "### equil done"; grep FINAL_L log.03_npt || true
