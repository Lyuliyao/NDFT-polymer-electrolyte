#!/bin/bash --login
# Field-on sampling for the release runs: array index i = potential p(i-1), runs/release_T1.0_c0.04/sample/pXX
#SBATCH --job-name=sip_rsm
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
P=$SIP_ROOT/runs/pilot_T1.0_c0.04_one_both
D=$SIP_ROOT/runs/release_T1.0_c0.04
DI=$SIP_ROOT/runs/drive_T1.0_c0.04/inputs
TAG=$(printf "p%02d" $((SLURM_ARRAY_TASK_ID - 1)))
mkdir -p $D/sample/$TAG && cd $D/sample/$TAG
# all 16 configurations present: done; otherwise restart from scratch (cheap, keeps cfg naming clean)
[ $(ls cfg.[0-9]* 2>/dev/null | wc -l) -ge 16 ] && { echo "### $TAG done"; exit 0; }
rm -f cfg.* log.sample
srun -n 32 $LMP -log log.sample -in $D/inputs/in.07_sample -var restart $P/field_$TAG/restart.field \
     -var ff $DI/ff.lmp -var tabfile $DI/solvation.table -var pppm_acc 1.0e-4 -var potfile $P/potentials/$TAG.lmp
check_ranks log.sample 32
