#!/bin/bash
# Size test field runs, chained on the build jobs.  (A) p01 p09 p13 p15, in-range
# wavelengths, the production run lengths of each eps_r at c = 0.04; (B) p30 p31,
# big-box m = 1 (k_1/n), n = 2, 4 only, transient 1 and production 5 x 1/(k^2 D_s).
# Dump every 100 n steps (n >= 2) so the data volume stays that of the small box;
# NREP keeps the profile block at 1e5 steps.
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer
declare -A BUILD=( [75_1]=18024530 [75_2]=18024531 [75_4]=18024532 [75_8]=18024533 [75_16]=18024534
                   [2_2]=18024535 [2_4]=18024536 [2_8]=18024537 [2_16]=18024538 )
declare -A NT=( [75]=1200000 [2]=3200000 ); declare -A NS=( [75]=11600000 [2]=15200000 )
declare -A DS=( [75]=2.80e-3 [2]=2.24e-3 )            # D_s at c = 0.04 (cation)
LS=( [75]=24.502285 [2]=24.416564 )
for key in "${!BUILD[@]}"; do
  e=${key%_*}; n=${key#*_}; S=$PWD/bigbox/eps${e}_c0.04_x$n
  [ $n = 1 ] && nd=200 || nd=$((100 * n)); nr=$((100000 / nd))
  common="FF=$S/ff.lmp TAB=$S/solvation.table SRCRESTART=$S/zero_field/restart.zf DEP=afterok:${BUILD[$key]}
          PPPM=1.0e-4 NTASKS=$((128 * n)) NNODES=$n MEMCPU=1G NRST=250000 NDUMP=$nd NREP=$nr
          PARTITION=scavenger ACCOUNT=scavenger QOS=scavenger"
  echo "== eps$e x$n (dump $nd, nrep $nr)"
  env $common ARRAY=1,9,13,15 TLIM=4-00:00:00 ./run_state.sh field $S ${NT[$e]} ${NS[$e]} | tail -1
  if [ $n = 2 ] || [ $n = 4 ]; then
    read ntr nst < <(python3 -c "
import math; L=$n*${LS[$e]}; t=(L/(2*math.pi))**2/${DS[$e]}; r=lambda x:int(round(x/0.005/2.5e5)*2.5e5)
print(r(t), r(5*t))")
    echo "   long-wavelength: tau(k1/$n) = $(python3 -c "print(f'{$ntr*0.005:.3g}')") tau -> ntrans $ntr, nsteps $nst"
    env $common ARRAY=30,31 TLIM=7-00:00:00 ./run_state.sh field $S $ntr $nst | tail -1
  fi
done
