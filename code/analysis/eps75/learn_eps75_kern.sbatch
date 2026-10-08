#!/bin/bash --login
# eps_r = 7.5 confirmation of the kernel basis chosen at eps_r = 2 (learned radial kernels B_m(r)[1 + g(r)], widest
# Gaussian 1 sigma, M = 6, softplus): the quoted V2 command (learn_seed.sbatch) with the basis changed, three seeds.
#   sbatch --export=ALL,SEED=0 learn_eps75_kern.sbatch
#SBATCH --job-name=sip_kern75
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps75/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps75
export SIP_ROOT=$R JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $R/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
: ${SEED:?}
SUF=_kn1softplus_long_s$SEED; LOG=$R/runs/learn/protocol_kn1softplus_s$SEED.log
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --kernel-net 32 32 --s-max 1 --M 6 --activation softplus \
    --joint-conc 0.01 0.02 0.04 0.06 0.08 --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $R/runs/learn/joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3$SUF \
    --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
