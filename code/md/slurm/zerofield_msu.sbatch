#!/bin/bash --login
# Stage 4: zero-field NVT production at the mean NPT volume.
# MSU HPCC (amd20) counterpart of zerofield.sbatch; run_state.sh submits it when SIP_SITE=msu.
#SBATCH --job-name=sip_zf
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lyuliyao@msu.edu
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU   # stale values from a submitting
# environment that is itself inside a SLURM job collide with --mem-per-cpu in srun
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${RUNDIR:?}; : ${LT:?}; : ${SRCRESTART:?}; : ${TEMP:=1.0}
: ${NEQUIL:=200000}; : ${NSTEPS:=4000000}; : ${NDUMP:=200}; : ${PPPM:=1.0e-4}
NP=${SLURM_NTASKS:-32}
mkdir -p $RUNDIR && cd $RUNDIR

srun -n $NP $LMP -log log.04_zerofield -in $SIP_ROOT/templates/in.04_zerofield \
       -var restart $SRCRESTART -var ff $SIP_ROOT/templates/ff.lmp \
       -var tabfile $SIP_ROOT/tools/solvation.table -var out restart.zf \
       -var T $TEMP -var Lt $LT -var nequil $NEQUIL -var nsteps $NSTEPS \
       -var ndump $NDUMP -var pppm_acc $PPPM
check_ranks log.04_zerofield $NP
echo "### zerofield done (dump left uncompressed for analysis)"
