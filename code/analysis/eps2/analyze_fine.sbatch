#!/bin/bash --login
# Grid-refinement test: re-bin every production field run at 0.05 sigma into the *_dz005 roots
# (tools/analyze_profiles_legacy.py, the error model of the paper models).  One task per line of fine_runs.txt.
#   sbatch --array=0-N%48 analyze_fine.sbatch
#SBATCH --job-name=sip_fine
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/fine_analysis/%A_%a.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
LIST=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/fine_runs.txt
read RUN POT MMAX OUT < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
echo "### $(date): $RUN -> $OUT (mmax $MMAX, binw 0.05)"
$SIP_PY $SIP_ROOT/tools/analyze_profiles_legacy.py --run $RUN --pot $POT --mmax $MMAX --binw 0.05 --out $OUT
echo "### done $(date)"
