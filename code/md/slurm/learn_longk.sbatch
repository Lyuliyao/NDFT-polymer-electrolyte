#!/bin/bash --login
# Variant of learn_long.sbatch with --kboost 2 (mode weight (k_3/k)^2 on m = 1, 2 of the long runs).
# After the long-wavelength field runs (m = 1, 2) are analysed: refresh the
# acceptance table, retrain on the five concentrations with the new runs
# included (experiment names get the suffix _longk), predict c = 0.05, and
# redraw the k-space diagnostic and the V1/V2 figure.  Submitted with
# --dependency=afterok:<analysis arrays> by the chain of 2026-09-25.
#SBATCH --job-name=sip_learn_longkk
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/learn/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
export JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
cd $SIP_ROOT
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|^Traceback|^  File |^    \^|cuda_device_count'
echo "### $(date): refresh the acceptance table"
cp runs/field_analysis/acceptance_summary.csv runs/field_analysis/acceptance_summary_before_longk.csv
$SIP_PY tools/acceptance_summary.py runs/pilot_T1.0_c0.04_one_both runs/prod_T1.0_c0.01 runs/prod_T1.0_c0.02 \
   runs/prod_T1.0_c0.05 runs/prod_T1.0_c0.06 runs/prod_T1.0_c0.08 --csv runs/field_analysis/acceptance_summary.csv | grep -E "^runs/|->"
C5="0.01 0.02 0.04 0.06 0.08"
echo "### $(date): V1 on five concentrations with the long runs"
$PY -m learn.protocol v1 --conc $C5 --name-suffix _longk --kboost 2 --log runs/learn/protocol.log 2>&1 | grep -vE "$F"
echo "### $(date): V2, V3 joint"
$PY -m learn.protocol joint --nval 3 --variants v2 v3 --depth 1 --C 4 --joint-conc $C5 --name-suffix _longk --kboost 2 --log runs/learn/protocol.log 2>&1 | grep -vE "$F"
for m in v1_c001_c002_c004_c006_c008_longk joint_v2_L1_C4_c001_c002_c004_c006_c008_val3_longk joint_v3_L1_C4_c001_c002_c004_c006_c008_val3_longk; do
  $PY -m learn.protocol predict --model runs/learn/$m --conc 0.05 --log runs/learn/protocol.log 2>&1 | grep -vE "$F"
done
echo "### $(date): figures"
$PY -m learn.diagnose_k --v1 v1_c001_c002_c004_c006_c008_longk --v2 joint_v2_L1_C4_c001_c002_c004_c006_c008_val3_longk --out runs/learn/figures/k_diagnostic_longk.png 2>&1 | grep -vE "$F"
$PY -m learn.present --v1 v1_c001_c002_c004_c006_c008_longk --v2 joint_v2_L1_C4_c001_c002_c004_c006_c008_val3_longk --out runs/learn/figures/v2_vs_v1_longk.png 2>&1 | grep -vE "$F"
$PY -m learn.protocol summary --log runs/learn/protocol.log > /dev/null 2>&1
echo "### done $(date)"
