#!/bin/bash --login
# Grid-refinement test: the paper's neural functional (learned radial kernels, _kn1softplus) trained on the
# production runs re-binned at 0.05 sigma (roots *_dz005; same splits, seeds and protocol as the paper models).
#   sbatch --export=ALL,ROOT=<dz005 root>,TAG=long|'',SEED=0 learn_fine.sbatch
#SBATCH --job-name=sip_kfine
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/fine_analysis/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
: ${ROOT:?}; : ${SEED:?}; TAG=${TAG:-}
KB=${KB:-0}; MODE=${MODE:-k}; KBS=$([ "$KB" != 0 ] && echo _kbK$KB); KBA=$([ "$KB" != 0 ] && echo "--kboost $KB --kboost-mode $MODE")
export SIP_ROOT=$ROOT JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $ROOT/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
SUF=_kn1softplus_dz005${KBS}${TAG:+_$TAG}_s$SEED
NAME=joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3$SUF
LOG=$ROOT/runs/learn/protocol_kn1softplus_dz005_s$SEED.log
n=$(ls $ROOT/runs/*/field_p*/profiles.npz | wc -l); echo "### $(date): $n fine profiles in $ROOT; seed $SEED -> $NAME"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --kernel-net 32 32 --s-max 1 --M 6 --activation softplus \
    --joint-conc 0.01 0.02 0.04 0.06 0.08 $KBA --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $ROOT/runs/learn/$NAME --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
