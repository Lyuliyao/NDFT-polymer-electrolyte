#!/bin/bash --login
# Size test: build a z-replicated box from a production zero-field restart.
#   sbatch -N <n> -n <128 n> --export=ALL,SRC=<small restart.zf>,FF=<ff.lmp>,TAB=<table>,NREP=<n>,OUT=<dir> ...
#SBATCH --job-name=sip_bigbuild
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --requeue
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${SRC:?}; : ${FF:?}; : ${TAB:?}; : ${NREP:?}; : ${OUT:?}; : ${NEQUIL:=200000}; : ${PPPM:=1.0e-4}; : ${LT:=0.0}
NP=$SLURM_NTASKS
Z=$OUT/zero_field; mkdir -p $Z; cd $Z
if [ ! -f small.data ]; then
  srun -n 32 $LMP -log log.restart2data -in $SIP_ROOT/templates/in.restart2data \
       -var restart $SRC -var ff $FF -var tabfile $TAB -var out small.data -var pppm_acc $PPPM
fi
srun -n $NP $LMP -log log.bigbox -in $SIP_ROOT/templates/in.bigbox \
     -var data small.data -var nrep $NREP -var ff $FF -var tabfile $TAB -var out restart.zf \
     -var nequil $NEQUIL -var Lt $LT -var seed $((20260929 + NREP)) -var pppm_acc $PPPM
check_ranks log.bigbox $NP
grep -h "Performance\|Loop time" log.bigbox
echo "### built $OUT ($(date))"
