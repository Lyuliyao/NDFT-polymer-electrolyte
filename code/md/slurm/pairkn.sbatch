#!/bin/bash --login
# Pair closure with the learned radial kernels of the paper's functional (2026-10-08, user: the pair closure uses the same
# learnable radial kernels, so that the comparison isolates the nonlinear term): variant v1 with kernel_net, trained by
# gradient exactly like the main model (force matching with the spectrum term, val3 early stopping, 4000 steps, seeds).
#   sbatch --export=ALL,ROOT=<root>,SEED=0,TAG=long|lj1|'',CONC="0.01 0.02 0.04 0.06 0.08",CT=c001_c002_c004_c006_c008,PRED=0.05[,NREF=0.4] slurm/pairkn.sbatch
#SBATCH --job-name=sip_pairkn
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/slurm/logs/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
: ${ROOT:?}; : ${SEED:?}; : ${CONC:?}; : ${CT:?}; TAG=${TAG:-}; PRED=${PRED:-}; NREF=${NREF:-}
export SIP_ROOT=$ROOT JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $ROOT/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
SUF=_kn1softplus_kbK2${TAG:+_$TAG}_s$SEED
NAME=joint_v1_L1_C4_R128x128_H64_${CT}_val3$SUF
LOG=$ROOT/runs/learn/protocol_pairkn_${CT}_s$SEED.log
echo "### $(date): pair closure, learned kernels, seed $SEED, concentrations $CONC -> $NAME"
$PY -m learn.protocol joint --nval 3 --variants v1 --depth 1 --C 4 --readout-hidden 128 128 ${NREF:+--n-ref $NREF} \
    --kernel-net 32 32 --s-max 1 --M 6 --activation softplus \
    --joint-conc $CONC --kboost 2 --kboost-mode k --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
if [ -n "$PRED" ]; then
  $PY -m learn.protocol predict --model $ROOT/runs/learn/$NAME --conc $PRED --log $LOG 2>&1 | grep -vE "$F"
fi
echo "### done $(date)"
