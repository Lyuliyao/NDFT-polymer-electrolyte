#!/bin/bash
# Size test with the C7 settings (coul cutoff 8, neighbor/comm multi, PPPM order 7, ik;
# verified 2026-09-29, runs/bench_pppm/V_*), scavenger on amd22|amd24, one node per run.
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer
export SLURM_HINT=nomultithread
SC="-p scavenger -A scavenger -q scavenger --constraint=amd22|amd24"
declare -A NT=( [75]=1200000 [2]=3200000 ); declare -A NS=( [75]=11600000 [2]=15200000 )
declare -A DS=( [75]=2.80e-3 [2]=2.24e-3 ); declare -A LS=( [75]=24.502285 [2]=24.416564 )
fields(){  # $1 state dir, $2 ranks, $3 dep, $4 ndump, $5 array, $6 ntrans, $7 nsteps, $8 src restart
  env FF=$1/ff.lmp TAB=$1/solvation.table SRCRESTART=$8 DEP=$3 PPPM=1.0e-4 NTASKS=$2 NNODES=1 MEMCPU=1G \
      NRST=250000 NDUMP=$4 NREP=$((100000 / $4)) PARTITION=scavenger ACCOUNT=scavenger QOS=scavenger \
      CONSTRAINT="amd22|amd24" TLIM=7-00:00:00 ARRAY=$5 ./run_state.sh field $1 $6 $7 | tail -1; }
for e in 75 2; do for n in 1 2 4 8; do
  [ $e = 2 ] && [ $n = 1 ] && continue
  S=$PWD/bigbox/eps${e}_c0.04_x$n
  if [ -f $S/zero_field/restart.zf ]; then DEP=""; echo "== eps$e x$n: existing build"; else
    if [ $e = 75 ]; then SRC=$PWD/runs/pilot_T1.0_c0.04_one_both/equil/restart.npt; LT=24.502285
    else SRC=$PWD/eps2/runs/prod_T1.0_c0.04/zero_field/restart.zf; LT=0.0; fi
    b=$(sbatch --parsable $SC -N 1 -n 64 --ntasks-per-node=64 --mem-per-cpu=1G --time=12:00:00 \
        -o $S/build.%j.out --export=ALL,SRC=$SRC,FF=$S/ff.lmp,TAB=$S/solvation.table,NREP=$n,OUT=$S,LT=$LT \
        slurm/bigbox_build_msu.sbatch)
    DEP=afterok:$b; echo "== eps$e x$n: build $b"
  fi
  [ $n = 1 ] && nd=200 || nd=$((100 * n))
  echo -n "   (A) "; fields $S 64 "$DEP" $nd 1,9,13,15 ${NT[$e]} ${NS[$e]} $S/zero_field/restart.zf
  if [ $n = 2 ] || [ $n = 4 ]; then
    read ntr nst < <(python3 -c "
import math; L=$n*${LS[$e]}; t=(L/(2*math.pi))**2/${DS[$e]}; r=lambda x:int(round(x/0.005/2.5e5)*2.5e5)
print(r(t), r(5*t))")
    [ $n = 4 ] && np=96 || np=64
    echo -n "   (B) $np ranks, ntrans $ntr nsteps $nst: "; fields $S $np "$DEP" $nd 30,31 $ntr $nst $S/zero_field/restart.zf
  fi
done; done
C=$PWD/bigbox/control_eps2_c0.04_C7
echo -n "== control eps2 c=0.04 p13 (C7, small box): "
fields $C 64 "" 200 13 ${NT[2]} ${NS[2]} $PWD/eps2/runs/prod_T1.0_c0.04/zero_field/restart.zf
