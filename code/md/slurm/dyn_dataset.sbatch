#!/bin/bash --login
# Dynamical training data (03h 6.11): one zero-field F(k,t) or one field-run C(m,m';t) per array task.
#   sbatch --array=1-N --export=ALL,MODE=cmm,LIST=runs/dyn_dataset/field_runs.txt slurm/dyn_dataset.sbatch
#SBATCH --job-name=sip_dyn
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
: ${MODE:?}; : ${LIST:?}
D=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SIP_ROOT/$LIST)
cd $SIP_ROOT
if [ "$MODE" = fkt ]; then $SIP_PY tools/dyn_dataset.py fkt --zf $D; else $SIP_PY tools/dyn_dataset.py cmm --run $D; fi
