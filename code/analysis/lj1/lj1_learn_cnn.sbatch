#!/bin/bash --login
# ONE LABEL LJ (2026-10-07): the convolutional free energy of Dijkman et al. (PRL 134, 056103, 2025; variant cnn: six periodic
# dilated convolutions, kernel 3, dilation 2, channels 16 16 32 32 64 64, average pooling by 2, linear read-out, softplus), trained
# exactly like the other LJ forms (lj1_learn.sbatch): force matching + spectrum term, val3, three seeds, nested fractions, n_ref 0.4.
# The 24 sigma box has 240 bins, which pool to the same 3 positions as the ion boxes.
#   SEED=0 [FRAC=0.25] sbatch --export=ALL lj1_learn_cnn.sbatch
#SBATCH --job-name=lj1_cnn
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj1/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj1
export SIP_ROOT=$R JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $R/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
: ${SEED:=0}; : ${FRAC:=1}; : ${STEPS:=4000}
CONC="0.1 0.2 0.4 0.6 0.7"; T=c01_c02_c04_c06_c07
FT=; [ "$FRAC" != 1 ] && FT=_f$(awk -v f=$FRAC 'BEGIN{printf "%03d", f*100+0.5}')
SUF=${FT}_kbK2_lj1_s$SEED; NAME=joint_cnn_d2_L1_C4_${T}_val3$SUF
LOG=$R/runs/learn/protocol_cnn${FT}_kbK2_s$SEED.log
echo "### $(date): cnn seed $SEED fraction $FRAC -> $NAME"
$PY -m learn.protocol joint --nval 3 --variants cnn --activation softplus --cnn-dilation 2 --depth 1 --C 4 --n-ref 0.4 \
    --joint-conc $CONC --train-frac $FRAC --kboost 2 --kboost-mode k --seed $SEED --steps $STEPS --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $R/runs/learn/$NAME --conc 0.5 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
