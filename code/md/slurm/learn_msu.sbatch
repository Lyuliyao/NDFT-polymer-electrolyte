#!/bin/bash --login
# Train / evaluate the neural density functional (learn/, doc/training_spec.md).
#   sbatch slurm/learn_msu.sbatch full
#   sbatch slurm/learn_msu.sbatch depth-scan --conc 0.04 --steps 4000
# Everything is CPU JAX on 1D grids; one node's cores are plenty.
#SBATCH --job-name=sip_learn
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
export SIP_ROOT=/mnt/gs21/scratch/lyuliyao/salt_in_polymer
export JAX_PLATFORMS=cpu
export PYTHONNOUSERSITE=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-4}"
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python      # jax 0.10 + optax + matplotlib
mkdir -p $SIP_ROOT/runs/learn
cd $SIP_ROOT
echo "### learn.protocol $*  $(date)  host $(hostname)"
$PY -m learn.protocol "$@" 2>&1 | grep -v "^Jax plugin\|jax_plugins\|cuInit\|versions_helpers\|_check_cuda\|plugin_module.initialize\|^Traceback\|^  File \|^    \^\|cuda_device_count"
echo "### done $(date)"
