#!/bin/bash --login
# 200 V2 training steps on a GPU, same data/config as the CPU runs, to compare
# the time per step (CPU: ~0.1 s/step from protocol.log).  Throwaway output.
#SBATCH --job-name=sip_gpubench
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/learn/%x-%j.out
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2
export SIP_ROOT=$R PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $R/code
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -c "import jax; print('devices', jax.devices()); import jax.numpy as jnp; print('x64', jax.config.jax_enable_x64)"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --joint-conc 0.01 0.02 0.04 0.06 --steps 200 --name-suffix _gpubench --log $R/runs/learn/protocol_gpubench.log 2>&1 \
    | grep -E "step|devices|->|Error" | head -20
