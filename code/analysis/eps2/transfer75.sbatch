#!/bin/bash --login
# eps_r transfer test: a model trained at eps_r = 7.5 (F_ex^theta unchanged; V1
# and V2 contain no l_B in F_ex) predicts the eps_r = 2 data, whose geometry
# carries l_B = 29.84 sigma into F_Coul.  Every eps_r = 2 run is unseen.
#   sbatch --export=ALL,MODEL=transfer75_v2_s0 transfer75.sbatch
#SBATCH --job-name=sip_xfer75
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
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
: ${MODEL:?}
for c in 0.01 0.02 0.04 0.05 0.06 0.08; do
  echo "### $(date): $MODEL on eps_r = 2, c = $c"
  $PY -m learn.protocol predict --model $R/runs/learn/$MODEL --conc $c --log $R/runs/learn/protocol_transfer75.log 2>&1 | grep -vE "$F" || echo "predict failed at c = $c"
done
echo "### done $(date)"
