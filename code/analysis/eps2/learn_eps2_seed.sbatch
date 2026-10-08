#!/bin/bash --login
# V2 with another initialisation seed (the validation split stays fixed), same
# data and configuration as learn_eps2.sbatch; V1 is a linear least-squares fit
# and does not depend on a seed.  Names carry the suffix _full_s<SEED>.
#   sbatch --export=ALL,SEED=1,TRAIN="0.01 0.02 0.04 0.06 0.08" learn_eps2_seed.sbatch
#SBATCH --job-name=sip_learn_seed
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
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
: ${SEED:?}; : ${TRAIN:="0.01 0.02 0.04 0.06 0.08"}
TT=$(for c in $TRAIN; do printf "_c%s" $(echo $c | tr -d .); done)
SUF=_full_s$SEED
LOG=$R/runs/learn/protocol_s$SEED.log
echo "### $(date): V2 seed $SEED on $TRAIN"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --joint-conc $TRAIN --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $R/runs/learn/joint_v2_L1_C4_R128x128_H64${TT}_val3$SUF --conc 0.05 \
    --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
