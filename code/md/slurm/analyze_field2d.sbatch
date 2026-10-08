#!/bin/bash --login
# Maps, mode series and acceptance of the two-dimensional field runs (tools/analyze_profiles2d.py).
#   sbatch --export=ALL,STATEDIR=<field2d/eps..._c0.04> --array=50-53 analyze_field2d.sbatch
#SBATCH --job-name=sip_f2da
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/field2d/%x-%A_%a.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
: ${STATEDIR:?}
TAG=p${SLURM_ARRAY_TASK_ID}
echo "### 2D analysis of $STATEDIR/field_$TAG  $(date)"
cd $STATEDIR
$SIP_PY $SIP_ROOT/tools/analyze_profiles2d.py --run field_$TAG --pot potentials/$TAG.json
echo "### done $(date)"
