#!/bin/bash --login
# First training on the eps_r = 2 campaign (before c = 0.08 is in): acceptance
# table -> V1 and the quoted V2 configuration (L = 1, C = 4, readout 128x128,
# 3 validation runs per c) on c = 0.01, 0.02, 0.04, 0.06 -> predict the
# untrained c = 0.05 -> Gamma from MD.  Everything runs with SIP_ROOT=eps2, so
# data, splits and runs/learn are the eps_r = 2 ones and the eps_r = 7.5
# results are not touched.
#SBATCH --job-name=sip_learn_eps2
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/eps2/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
CODE=$SIP_ROOT
export SIP_ROOT=$CODE/eps2 JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $CODE                      # the learn package is imported from here
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
: ${TRAIN:="0.01 0.02 0.04 0.06"}
TT=$(for c in $TRAIN; do printf "_c%s" $(echo $c | tr -d .); done)
LOG=$SIP_ROOT/runs/learn/protocol.log
echo "### $(date): acceptance table (eps_r = 2)"
S=""; for c in 0.01 0.02 0.04 0.05 0.06 0.08; do
  d=eps2/runs/prod_T1.0_c$c; ls $d/field_p*/acceptance.json >/dev/null 2>&1 && S="$S $d"; done
$SIP_PY tools/acceptance_summary.py $S --csv eps2/runs/field_analysis/acceptance_summary.csv | grep -E "^eps2/|->"
$PY -c "import learn.data as D; print('### learn root', D.ROOT); [print(c, sp.path, 'lB=%.3f' % sp.lB) for c, sp in sorted(D.discover_state_points().items())]" 2>&1 | grep -vE "$F"
echo "### $(date): V1"
$PY -m learn.protocol v1 --conc $TRAIN --log $LOG 2>&1 | grep -vE "$F"
echo "### $(date): V2 (quoted configuration)"
$PY -m learn.protocol joint --nval 3 --variants v2 --depth 1 --C 4 --readout-hidden 128 128 \
    --joint-conc $TRAIN --log $LOG 2>&1 | grep -vE "$F"
for m in v1$TT joint_v2_L1_C4_R128x128_H64${TT}_val3; do
  echo "### $(date): predict c = 0.05 with $m"
  $PY -m learn.protocol predict --model $SIP_ROOT/runs/learn/$m --conc 0.05 --log $LOG 2>&1 | grep -vE "$F"
done
echo "### $(date): Gamma from MD"
$PY -m learn.gamma_md --concs 0.01 0.02 0.04 0.05 0.06 2>&1 | grep -vE "$F" || echo "gamma_md failed (non-fatal)"
$PY -m learn.protocol summary --log $LOG > /dev/null 2>&1 || true
echo "### done $(date)"
