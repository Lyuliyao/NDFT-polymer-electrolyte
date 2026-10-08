#!/bin/bash
# Size test on data-machine (user, 2026-09-29): builds + field runs.  PPPM-bound,
# so ranks saturate early (4x box: 96 ranks 249 steps/s, 192 ranks 257 on Zen4):
# 64 ranks for x1, 128 for x2-x8 (one node), 256 for x16 (two nodes); physical
# cores only (acm/nal nodes are 128 cores x 2 threads).
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer
export SLURM_HINT=nomultithread
DM="-p data-machine -A data-machine --constraint=amd22|amd24"
declare -A NT=( [75]=1200000 [2]=3200000 ); declare -A NS=( [75]=11600000 [2]=15200000 )
declare -A DS=( [75]=2.80e-3 [2]=2.24e-3 ); declare -A LS=( [75]=24.502285 [2]=24.416564 )
for e in 75 2; do for n in 1 2 4 8 16; do
  [ $e = 2 ] && [ $n = 1 ] && continue
  S=$PWD/bigbox/eps${e}_c0.04_x$n
  case $n in 1) R=64; N=1;; 16) R=256; N=2;; *) R=128; N=1;; esac
  if [ -f $S/zero_field/restart.zf ]; then DEP=""; else
    if [ $e = 75 ]; then SRC=$PWD/runs/pilot_T1.0_c0.04_one_both/equil/restart.npt; LT=24.502285
    else SRC=$PWD/eps2/runs/prod_T1.0_c0.04/zero_field/restart.zf; LT=0.0; fi
    b=$(sbatch --parsable $DM -N $N -n $R --ntasks-per-node=$((R / N)) --mem-per-cpu=1G --time=12:00:00 \
        -o $S/build.%j.out --export=ALL,SRC=$SRC,FF=$S/ff.lmp,TAB=$S/solvation.table,NREP=$n,OUT=$S,LT=$LT \
        slurm/bigbox_build_msu.sbatch)
    DEP=afterok:$b; echo "== eps$e x$n: build $b ($R ranks on $N node)"
  fi
  [ $n = 1 ] && nd=200 || nd=$((100 * n)); nr=$((100000 / nd))
  common="FF=$S/ff.lmp TAB=$S/solvation.table SRCRESTART=$S/zero_field/restart.zf DEP=$DEP
          PPPM=1.0e-4 NTASKS=$R NNODES=$N MEMCPU=1G NRST=250000 NDUMP=$nd NREP=$nr
          PARTITION=data-machine ACCOUNT=data-machine CONSTRAINT=amd22|amd24 TLIM=7-00:00:00"
  echo "   (A) eps$e x$n: dump $nd, nrep $nr"
  env $common ARRAY=1,9,13,15 ./run_state.sh field $S ${NT[$e]} ${NS[$e]} | tail -1
  if [ $n = 2 ] || [ $n = 4 ]; then
    read ntr nst < <(python3 -c "
import math; L=$n*${LS[$e]}; t=(L/(2*math.pi))**2/${DS[$e]}; r=lambda x:int(round(x/0.005/2.5e5)*2.5e5)
print(r(t), r(5*t))")
    echo "   (B) eps$e x$n: ntrans $ntr, nsteps $nst"
    env $common ARRAY=30,31 ./run_state.sh field $S $ntr $nst | tail -1
  fi
done; done
