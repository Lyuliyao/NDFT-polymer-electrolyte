#!/bin/bash --login
# eps_r = 2, the quoted V2 configuration with a different kernel basis; same data (the 82 training runs of the
# _lin models), validation, protocol and three seeds.
#   KTAG=smax4 ARCH="--s-max 4 --M 10"   widest Gaussian 4 sigma (same log spacing)
#   KTAG=smax8 ARCH="--s-max 8 --M 13"   widest Gaussian 8 sigma
#   KTAG=knet  ARCH="--kernel-net 32 32" learned radial kernels B_m(r)[1 + g(r)]
#   sbatch --export=ALL,KTAG=knet,ARCH="--kernel-net 32 32",SEED=0 learn_eps2_kern.sbatch
#SBATCH --job-name=sip_kern2
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
: ${KTAG:?}; : ${ARCH:?}; : ${SEED:=0}
LOG=$R/runs/learn/protocol_${KTAG}_s$SEED.log
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 $ARCH \
    --joint-conc 0.01 0.02 0.04 0.06 0.08 --seed $SEED --name-suffix _${KTAG}_s$SEED --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $R/runs/learn/joint_v2_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3_${KTAG}_s$SEED \
    --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
