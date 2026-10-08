#!/bin/bash --login
# Size test: every model of this eps_r predicts the big-box runs of one root (EL
# profiles from V(z) alone, force residual, bulk structure).  Nothing is trained.
#   sbatch --export=ALL,ROOT=<root> predict.sbatch
#SBATCH --job-name=sip_bbpred
#SBATCH --account=multiscaleml
#SBATCH --partition=general-short
#SBATCH --constraint=amd24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:59:00
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox/%x-%j.out
set -e
: ${ROOT:?}
export SIP_ROOT=$ROOT JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $ROOT/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
: ${MODELS:="V1 V2_s0 V2_s1 V2_s2"}
for m in $MODELS; do
  echo "### $(date): $m on $(basename $ROOT)"
  $PY -m learn.protocol predict --model $ROOT/runs/learn/$m --conc 0.04 ${PREDICT_ARGS:-} --log $ROOT/runs/learn/protocol.log 2>&1 | grep -vE "$F"
done
echo "### done $(date)"
