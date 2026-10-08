#!/bin/bash --login
#SBATCH --job-name=sip_fz2
#SBATCH --account=multiscaleml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
cd $SIP_ROOT; $SIP_PY tools/dyn_fz2.py $(sed -n "${SLURM_ARRAY_TASK_ID}p" runs/dyn_dataset/fz2_list.txt)
