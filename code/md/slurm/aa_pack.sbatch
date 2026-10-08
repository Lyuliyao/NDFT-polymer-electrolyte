#!/bin/bash --login
#SBATCH --job-name=aa_pack
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/prod_r004/%x-%j.out
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/prod_r004
export PATH=/mnt/home/lyuliyao/.conda/envs/aamd/bin:$PATH
date; packmol < pack.inp > packmol.log; date
grep -E "Success|ENDED WITHOUT" packmol.log
