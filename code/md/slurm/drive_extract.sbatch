#!/bin/bash --login
#SBATCH --job-name=sip_dex
#SBATCH --account=multiscaleml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
cd $SIP_ROOT; $SIP_PY tools/drive_extract.py --nproc 16
