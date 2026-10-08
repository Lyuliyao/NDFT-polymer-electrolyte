#!/bin/bash --login
# Comparison of architectures (learn.models): the functional forms of earlier
# neural functionals, trained exactly like the quoted V2 (force matching, the
# analytic Coulomb part, five concentrations, val3 early stopping), then the
# untrained c = 0.05.
#   VARIANT=c1win ARCH="--win-bins 30"  one-body window network (Sammuller 2023; Bui & Cox 2025)
#   VARIANT=cace  ARCH="--cace-qcut 3"  equivariant local free energy (Cheng 2026, Equi-cDFT)
#   sbatch --export=ALL,ROOT=<eps root>,VARIANT=cace,ARCH="--cace-qcut 3",SEED=0,TAG=full learn_arch.sbatch
#SBATCH --job-name=sip_arch
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
: ${ROOT:?}; : ${VARIANT:?}; : ${ARCH:?}; : ${SEED:?}; : ${TAG:?}; : ${FRAC:=1}
KB=${KB:-0}; MODE=${MODE:-k}; KBS=$([ "$KB" != 0 ] && echo _kbK$KB); KBA=$([ "$KB" != 0 ] && echo "--kboost $KB --kboost-mode $MODE")
export SIP_ROOT=$ROOT JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $ROOT/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
case "$VARIANT" in
  c1win) A=_W$(echo $ARCH | awk '{print $2}') ;;
  cace)  A=_a5q$(echo $ARCH | awk '{print $2}') ;;
  v2)    A= ;;
  *) echo "unknown VARIANT $VARIANT"; exit 1 ;;
esac
FT=; [ "$FRAC" != 1 ] && FT=_f$(printf '%03.0f' $(echo "$FRAC*100" | bc))
SUF=${FT}${KBS}_${TAG}_s$SEED; NAME=joint_${VARIANT}${A}_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3$SUF
LOG=$ROOT/runs/learn/protocol_arch_${VARIANT}${A}${FT}${KBS}_s$SEED.log
echo "### $(date): $VARIANT $ARCH seed $SEED fraction $FRAC root $ROOT -> $NAME"
$PY -m learn.protocol joint --nval 3 --variants $VARIANT $ARCH --depth 1 --C 4 --readout-hidden 128 128 \
    --joint-conc 0.01 0.02 0.04 0.06 0.08 --train-frac $FRAC $KBA --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $ROOT/runs/learn/$NAME --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
