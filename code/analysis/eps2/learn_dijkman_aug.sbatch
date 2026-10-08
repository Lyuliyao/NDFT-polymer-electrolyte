#!/bin/bash --login
# Symmetry by data: the same with NSH (default 7) random shifts and the mirror image of every copy of each training run.
# Dijkman et al. (PRL 134, 056103, 2025) architecture against ours, each with both training signals.
#   ARCH = cnn: their global CNN free energy (variant cnn: six periodic dilated convolutions, kernel 3,
#               dilation 2, channels 16 16 32 32 64 64, average pooling by 2, linear read-out; two ion
#               channels; analytic Coulomb term as for every model here)
#   ARCH = kn : the paper's neural functional (learned radial kernels, the _kn1softplus models)
#   LOSS = force (Eq. 7 of the paper) | pcm (pair-correlation matching on the zero-field S_ab(k))
# Same splits, protocol and early stopping as the other comparisons; then the untrained c = 0.05.
#   sbatch --export=ALL,ROOT=<eps root>,ARCH=cnn,LOSS=pcm,SEED=0,TAG=long|'' learn_dijkman.sbatch
#SBATCH --job-name=sip_dijkA
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
: ${ROOT:?}; : ${ARCH:?}; : ${LOSS:?}; : ${SEED:?}; TAG=${TAG:-}; : ${NSH:=7}
KB=${KB:-0}; MODE=${MODE:-k}; KBS=$([ "$KB" != 0 ] && echo _kbK$KB); KBA=$([ "$KB" != 0 ] && echo "--kboost $KB --kboost-mode $MODE")
export SIP_ROOT=$ROOT JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $ROOT/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
T=c001_c002_c004_c006_c008
case "$ARCH" in
  cnn) A="--variants cnn --activation softplus --cnn-dilation 2"; SUF=_${LOSS}_aug${NSH}R${KBS}${TAG:+_$TAG}_s$SEED
       NAME=joint_cnn_d2_L1_C4_${T}_val3$SUF ;;
  kn)  A="--variants v2 --readout-hidden 128 128 --kernel-net 32 32 --s-max 1 --M 6 --activation softplus"
       SUF=_kn1softplus_${LOSS}${KBS}${TAG:+_$TAG}_s$SEED; NAME=joint_v2_L1_C4_R128x128_H64_${T}_val3$SUF ;;
  *) echo "unknown ARCH $ARCH"; exit 1 ;;
esac
LOG=$ROOT/runs/learn/protocol_dijk_${ARCH}_${LOSS}_aug_s$SEED.log
echo "### $(date): $ARCH trained with $LOSS, seed $SEED, root $ROOT -> $NAME"
$PY -m learn.protocol joint --nval 3 $A --depth 1 --C 4 --joint-conc 0.01 0.02 0.04 0.06 0.08 \
    --loss $LOSS --augment-shifts $NSH --augment-reflect $KBA --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $ROOT/runs/learn/$NAME --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
