#!/bin/bash --login
# 2D comparison with the pair closure of learned kernels (its cached maps were removed; the other models stay cached).
#SBATCH --job-name=sip_f2dpair
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/slurm/logs/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
export JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer
$PY /mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/figs/field2d_compare.py 2>&1 | grep -vE "jax_plugins|cuInit|plugin|versions_helpers|_check_cuda|^Traceback|^  File |^    \^|cuda_device_count"
echo "### done $(date)"
