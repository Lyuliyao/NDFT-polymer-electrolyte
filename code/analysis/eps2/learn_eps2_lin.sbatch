#!/bin/bash --login
# eps_r = 2 with the two linear-regime runs at c = 0.01 (p40, p41: m = 1, 0.03 and 0.05 kT) added to
# training; everything else as the quoted models (the validation draw is unchanged: linear runs stay in
# training, learn.data.validation_split).  Names carry _lin_s<SEED> (V2) and _lin (V1).
#   sbatch --export=ALL,MODEL=v2,SEED=0 learn_eps2_lin.sbatch      MODEL=v1 for the pair closure
#SBATCH --job-name=sip_lin2
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
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2
export SIP_ROOT=$R JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $R/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
: ${MODEL:?}; : ${SEED:=0}
TRAIN="0.01 0.02 0.04 0.06 0.08"; T=_c001_c002_c004_c006_c008
if [ "$MODEL" = v1 ]; then
  LOG=$R/runs/learn/protocol_lin_v1.log
  $PY -m learn.protocol v1 --conc $TRAIN --name-suffix _lin --log $LOG 2>&1 | grep -vE "$F"
  NAME=v1${T}_lin
else
  LOG=$R/runs/learn/protocol_lin_s$SEED.log
  $PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
      --joint-conc $TRAIN --seed $SEED --name-suffix _lin_s$SEED --log $LOG 2>&1 | grep -vE "$F"
  NAME=joint_v2_L1_C4_R128x128_H64${T}_val3_lin_s$SEED
fi
$PY -m learn.protocol predict --model $R/runs/learn/$NAME --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
