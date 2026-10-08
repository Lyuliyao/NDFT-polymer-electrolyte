#!/bin/bash --login
# Kernel range: the paper's functional with the widest Gaussian at SMAX sigma and M widths (the eps_r = 2 kernel-scan
# variants kn2softplus: SMAX 2, M 8; kn3softplus: SMAX 3, M 9), otherwise as the paper models.
#   sbatch --export=ALL,ROOT=<eps root>,TAG=long|'',SEED=0,SMAX=2,M=8[,KB=2] learn_kern_range.sbatch   (KB: long-wavelength weighting as learn_kboost.sbatch)
#   sbatch --export=ALL,ROOT=<eps root>,TAG=long|'',SEED=0,FRAC=0.25 learn_kern_frac.sbatch
#SBATCH --job-name=sip_krange
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
: ${ROOT:?}; : ${SEED:?}; : ${SMAX:?}; : ${M:?}; TAG=${TAG:-}; KB=${KB:-0}
export SIP_ROOT=$ROOT JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $ROOT/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
FT=$([ "$KB" != 0 ] && echo _kb$KB)
SUF=_kn${SMAX}softplus${FT}${TAG:+_$TAG}_s$SEED
NAME=joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3$SUF
LOG=$ROOT/runs/learn/protocol_kn${SMAX}softplus${FT}_s$SEED.log
echo "### $(date): kn${SMAX}softplus (widest kernel $SMAX sigma, M $M) seed $SEED root $ROOT -> $NAME"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --kernel-net 32 32 --s-max $SMAX --M $M --activation softplus \
    --joint-conc 0.01 0.02 0.04 0.06 0.08 --kboost $KB --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $ROOT/runs/learn/$NAME --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
