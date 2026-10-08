#!/bin/bash --login
# eps_r = 2, five concentrations: V2 with re-convolution layers (depth L >= 2, C gated
# channels per layer), otherwise the quoted configuration (readout 128x128, 3
# validation runs per c).  Names: joint_v2_L<L>_C<C>_R128x128_H64_<c>_val3_full_s<SEED>.
#   sbatch --export=ALL,L=2,C=4,SEED=0 learn_eps2_deep.sbatch
#SBATCH --job-name=sip_learn_deep
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2
export SIP_ROOT=$R JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $R/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
: ${L:?}; : ${C:?}; : ${SEED:?}
TRAIN="0.01 0.02 0.04 0.06 0.08"; TT=_c001_c002_c004_c006_c008; SUF=_full_s$SEED
LOG=$R/runs/learn/protocol_L${L}_C${C}_s$SEED.log
echo "### $(date): V2 L=$L C=$C seed $SEED"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth $L --C $C --readout-hidden 128 128 \
    --joint-conc $TRAIN --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $R/runs/learn/joint_v2_L${L}_C${C}_R128x128_H64${TT}_val3$SUF --conc 0.05 \
    --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
