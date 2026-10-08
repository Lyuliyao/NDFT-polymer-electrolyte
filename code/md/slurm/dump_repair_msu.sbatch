#!/bin/bash --login
# Repair (MODE=repair) or only check (MODE=check) the ion dumps of one field run
# per array task; LIST holds lines "<conc>/<tag>" under $STATEROOT.
# See tools/dump_repair.py (the gs21 I/O fault of 2026-09-27).
#SBATCH --job-name=sip_dfix
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
set -e
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
: ${LIST:?}; : ${MODE:=check}; : ${STATEROOT:=$SIP_ROOT/eps2/runs}
L=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $LIST)
RUN=$STATEROOT/prod_T1.0_c${L%/*}/field_${L#*/}
FLAG=""; [ "$MODE" = check ] && FLAG=--check
cd $RUN
# gs21 returns ENOSPC in bursts although quota and df have room: retry
for try in 1 2 3; do
  $SIP_PY $SIP_ROOT/tools/dump_repair.py $FLAG cat.dump ani.dump && break
  [ $try = 3 ] && { echo "### giving up on $L"; exit 1; }
  echo "### attempt $try failed, retrying in 10 min"; sleep 600
done
# a repaired dump invalidates any .npz cache built from the damaged file
[ "$MODE" = repair ] && rm -f cat.dump*.npz ani.dump*.npz
echo "### $MODE $L done"
