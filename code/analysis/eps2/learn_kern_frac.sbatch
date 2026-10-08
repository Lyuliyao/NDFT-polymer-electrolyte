#!/bin/bash --login
# Training-set size for the neural functional of the paper (learned radial kernels B_m(r)[1 + g(r)], widest 1 sigma,
# M = 6, softplus; the _kn1softplus models): the quoted command with --train-frac, nested stratified fractions of the
# training runs (validation and held-out runs unchanged), three seeds; same as submit_arch_frac.sh for the baselines.
#   sbatch --export=ALL,ROOT=<eps root>,TAG=long|'',SEED=0,FRAC=0.25 learn_kern_frac.sbatch
#SBATCH --job-name=sip_kfrac
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
: ${ROOT:?}; : ${SEED:?}; : ${FRAC:?}; TAG=${TAG:-}
KB=${KB:-0}; MODE=${MODE:-k}; KBS=$([ "$KB" != 0 ] && echo _kbK$KB); KBA=$([ "$KB" != 0 ] && echo "--kboost $KB --kboost-mode $MODE")
export SIP_ROOT=$ROOT JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $ROOT/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
FT=_f$(printf '%03.0f' $(echo "$FRAC*100" | bc))
SUF=_kn1softplus${FT}${KBS}${TAG:+_$TAG}_s$SEED
NAME=joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3$SUF
LOG=$ROOT/runs/learn/protocol_kn1softplus${FT}${KBS}_s$SEED.log
echo "### $(date): kn1softplus fraction $FRAC seed $SEED root $ROOT -> $NAME"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --kernel-net 32 32 --s-max 1 --M 6 --activation softplus \
    --joint-conc 0.01 0.02 0.04 0.06 0.08 --train-frac $FRAC $KBA --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $ROOT/runs/learn/$NAME --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
