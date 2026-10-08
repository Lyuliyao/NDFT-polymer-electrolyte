#!/bin/bash --login
# eps_r = 2, first training, run entirely from research (gs21 scratch returns
# ENOSPC since 2026-09-27; the MD data there are only read, through the
# symlinks in runs/).  Train V1 and V2 (L1 C4 R128x128 val3) on c = 0.01, 0.02,
# 0.04, 0.06, predict the untrained c = 0.05, Gamma from MD.
#SBATCH --job-name=sip_learn_eps2
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2
export SIP_ROOT=$R JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $R/code
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
: ${TRAIN:="0.01 0.02 0.04 0.06"}
TT=$(for c in $TRAIN; do printf "_c%s" $(echo $c | tr -d .); done)
echo "### $(date): acceptance table"
S=""; for d in $R/runs/prod_T1.0_c*; do ls $d/field_p*/acceptance.json >/dev/null 2>&1 && S="$S $d"; done
/mnt/home/lyuliyao/.conda/envs/mdmd/bin/python $R/code/tools/acceptance_summary.py $S \
    --csv $R/runs/field_analysis/acceptance_summary.csv | grep -E "^/|->"
: ${SUF:=}                    # name suffix, e.g. _full once the damaged runs are back
LOG=$R/runs/learn/protocol.log
echo "### $(date): V1"
$PY -m learn.protocol v1 --conc $TRAIN --name-suffix "$SUF" --log $LOG 2>&1 | grep -vE "$F"
echo "### $(date): V2 (quoted configuration)"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --joint-conc $TRAIN --name-suffix "$SUF" --log $LOG 2>&1 | grep -vE "$F"
for m in v1$TT$SUF joint_v2_L1_C4_R128x128_H64${TT}_val3$SUF; do
  echo "### $(date): predict c = 0.05 with $m"
  $PY -m learn.protocol predict --model $R/runs/learn/$m --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
done
echo "### $(date): Gamma from MD"
$PY -m learn.gamma_md --concs 0.01 0.02 0.04 0.05 0.06 0.08 2>&1 | grep -vE "$F" || echo "gamma_md failed (non-fatal)"
$PY -m learn.protocol summary --log $LOG > /dev/null 2>&1 || true
echo "### done $(date)"
