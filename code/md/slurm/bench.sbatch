#!/bin/bash
#SBATCH -J sip_bench
#SBATCH -A mth210003p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 00:30:00
#SBATCH -o bench.%j.out
set -e
source /ocean/projects/mth210003p/lyuliyao/salt_in_polymer/env.sh
WORK=$SIP_ROOT/runs/bench && mkdir -p $WORK && cd $WORK

ranks() { grep -m1 -oE "[0-9]+ MPI tasks" "$1" | awk '{print $1}'; }
perf()  { grep -m1 -oE "[0-9.]+ katom-step/s" "$1" | awk '{print $1}'; }

echo "########## launcher check (want 8 ranks) ##########"
for L in "srun -n 8" "mpirun -n 8"; do
  $L $LMP -log lt.log -in $SIP_ROOT/templates/in.02_warmup \
     -var restart restart.pushoff -var ff $SIP_ROOT/templates/ff.lmp \
     -var tabfile $SIP_ROOT/tools/solvation.table -var out /dev/null \
     -var ncap 20 -var nnvt 20 > /dev/null 2>&1 || true
  echo "  '$L' -> $(ranks lt.log) MPI ranks"
done

echo "########## scaling with mpirun ##########"
for NP in 1 2 4 8 16 32; do
  mpirun -n $NP $LMP -log sc.$NP.log -in $SIP_ROOT/templates/in.02_warmup \
     -var restart restart.pushoff -var ff $SIP_ROOT/templates/ff.lmp \
     -var tabfile $SIP_ROOT/tools/solvation.table -var out /dev/null \
     -var ncap 20 -var nnvt 2000 > /dev/null 2>&1
  echo "NP=$NP ranks=$(ranks sc.$NP.log) katom_step_per_s=$(perf sc.$NP.log)"
done
echo "########## DONE ##########"
