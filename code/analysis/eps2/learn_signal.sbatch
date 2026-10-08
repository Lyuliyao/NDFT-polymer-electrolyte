#!/bin/bash --login
# Comparison of training signals (learn.signals): the quoted V2 architecture trained
# with LOSS = lmu (local chemical-potential balance, Cheng 2026) or pcm
# (pair-correlation matching on the zero-field S_ab(k), Dijkman 2025), five
# concentrations, then the untrained c = 0.05.
#   sbatch --export=ALL,ROOT=<eps root>,LOSS=pcm,SEED=0,TAG=full learn_signal.sbatch
#SBATCH --job-name=sip_signal
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
: ${ROOT:?}; : ${LOSS:?}; : ${SEED:?}; : ${TAG:?}
export SIP_ROOT=$ROOT JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $ROOT/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
SUF=_${LOSS}_${TAG}_s$SEED; LOG=$ROOT/runs/learn/protocol_signal_${LOSS}_s$SEED.log
echo "### $(date): V2 L1 trained with $LOSS, seed $SEED, root $ROOT"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --joint-conc 0.01 0.02 0.04 0.06 0.08 --loss $LOSS --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $ROOT/runs/learn/joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3$SUF \
    --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
