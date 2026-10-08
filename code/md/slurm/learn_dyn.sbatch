#!/bin/bash --login
# End-to-end mobility training (03h §7): sbatch --export=ALL,ARGS="--variant B --naux 3" slurm/learn_dyn.sbatch
#SBATCH --job-name=sip_ldyn
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer
export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16" OMP_NUM_THREADS=16
/mnt/home/lyuliyao/.conda/envs/heat/bin/python -m learn.dyn_train $ARGS 2>&1 | grep --line-buffered -v "WARNING\|I0000\|W0000"
