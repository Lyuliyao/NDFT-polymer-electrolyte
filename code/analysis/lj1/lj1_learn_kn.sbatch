#!/bin/bash --login
# ONE LABEL (2026-10-05): the one-component LJ fluid, the two MD labels merged (tools/lj_merge_labels.py), root salt_in_polymer_lj1, suffix _lj1.
# LJ benchmark with the neural functional of the paper (learned radial kernels B_m(r)[1 + g(r)], widest 1 sigma,
# M = 6, softplus; the _kn1softplus models), trained exactly like the ion models and like lj_learn.sbatch:
# force matching, val3 early stopping, three seeds, nested fractions of the training runs; n_ref = 0.4.
#   SEED=0 [FRAC=0.25] sbatch --export=ALL lj_learn_kn.sbatch
#SBATCH --job-name=lj1_kn
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj1/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj1
export SIP_ROOT=$R JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $R/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
: ${SEED:=0}; : ${FRAC:=1}
KB=${KB:-0}; MODE=${MODE:-k}; KBS=$([ "$KB" != 0 ] && echo _kbK$KB); KBA=$([ "$KB" != 0 ] && echo "--kboost $KB --kboost-mode $MODE")
CONC="0.1 0.2 0.4 0.6 0.7"; T=c01_c02_c04_c06_c07
FT=; [ "$FRAC" != 1 ] && FT=_f$(awk -v f=$FRAC 'BEGIN{printf "%03d", f*100+0.5}')
SUF=_kn1softplus${FT}${KBS}_lj1_s$SEED; NAME=joint_v2_L1_C4_R128x128_H64_${T}_val3$SUF
LOG=$R/runs/learn/protocol_kn1softplus${FT}${KBS}_s$SEED.log
echo "### $(date): kn1softplus seed $SEED fraction $FRAC -> $NAME"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 --n-ref 0.4 \
    --kernel-net 32 32 --s-max 1 --M 6 --activation softplus \
    --joint-conc $CONC --train-frac $FRAC $KBA --seed $SEED --name-suffix $SUF --log $LOG 2>&1 | grep -vE "$F"
$PY -m learn.protocol predict --model $R/runs/learn/$NAME --conc 0.5 --log $LOG 2>&1 | grep -vE "$F"
echo "### done $(date)"
