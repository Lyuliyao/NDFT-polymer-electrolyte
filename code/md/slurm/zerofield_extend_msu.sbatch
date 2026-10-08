#!/bin/bash --login
# Extend a finished zero-field run so that tau_int(m=3) can be measured
# directly instead of being inferred from D_s.
#   sbatch --export=ALL,STATE=runs/prod_T1.0_c0.08,TARGET=12000000 ...
#SBATCH --job-name=sip_zfx
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --open-mode=append
#SBATCH --requeue
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${STATE:?}; : ${TARGET:?}; : ${CONC:?}; : ${TEMP:=1.0}; : ${PPPM:=1.0e-4}
# force field, table and dielectric of the state point (defaults: the eps_r = 7.5 production model);
# an eps_r = 2 state point passes FF=$STATE/ff.lmp TAB=$STATE/solvation.table EPS_R=2
: ${FF:=$SIP_ROOT/templates/ff.lmp}; : ${TAB:=$SIP_ROOT/tools/solvation.table}; : ${EPS_R:=7.5}
grep -q "^dielectric *$EPS_R\$" $FF || { echo "FATAL: $FF is not dielectric $EPS_R"; exit 1; }
NP=${SLURM_NTASKS:-32}
Z=$STATE/zero_field
cd $Z

# a checkpoint cut mid-write (gs21 I/O faults, 2026-09-27) has an odd size: drop it
if ls restart.zf.[0-9]* >/dev/null 2>&1; then
  SZ=$(stat -c %s restart.zf.[0-9]* | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
  for f in restart.zf.[0-9]*; do [ "$(stat -c %s $f)" = "$SZ" ] || { echo "### dropping truncated $f"; rm -f $f; }; done
fi
# resume point: the newest periodic checkpoint, else the end of the first run
LAST=$(ls -1 restart.zf.[0-9]* 2>/dev/null | sed 's/.*restart\.zf\.//' | sort -n | tail -1)
if [ -n "$LAST" ]; then
  START=restart.zf.$LAST; SEG=$(ls msd.*.dat 2>/dev/null | wc -l)
  ls restart.zf.[0-9]* | grep -v "^restart\.zf\.$LAST$" | xargs -r rm -f
else
  START=restart.zf
  LAST=$($SIP_PY -c "
import sys; sys.path.insert(0,'$SIP_ROOT/tools')
print(open('ions.dump').read().count('ITEM: TIMESTEP') and 0)" 2>/dev/null || echo 0)
  # the first run ended where its last dump frame sits
  LAST=$(tail -c 400000 ions.dump | grep -A1 "ITEM: TIMESTEP" | tail -1)
  SEG=1
fi
echo "### extending $Z from step $LAST to $TARGET (segment $SEG), $(date)"
$SIP_PY $SIP_ROOT/tools/dump_truncate.py ions.dump $LAST

srun -n $NP $LMP -log log.04b_extend.$SEG -in $SIP_ROOT/templates/in.04b_extend \
     -var restart $START -var ff $FF \
     -var tabfile $TAB -var out restart.zf \
     -var T $TEMP -var target $TARGET -var seg $SEG -var pppm_acc $PPPM
check_ranks log.04b_extend.$SEG $NP
test -f restart.zf && rm -f restart.zf.[0-9]*

$SIP_PY $SIP_ROOT/tools/zero_field_report.py --run $Z --conc $CONC --temp $TEMP --eps-r $EPS_R \
        | tee $STATE/zero_field_report.txt
# learn/ takes l_B from Sk.npz: it must carry this state point's eps_r
$SIP_PY -c "
import numpy as np, sys; sys.path.insert(0, '$SIP_ROOT/tools'); import model as M
lb = float(np.load('$Z/Sk.npz')['lB']); ref = M.bjerrum_sigma($TEMP, $EPS_R)
assert abs(lb - ref) < 1e-9, (lb, ref); print(f'### l_B in Sk.npz = {lb:.4f} (eps_r = $EPS_R)')"
echo "### extension done $(date)"
set +e; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State; exit 0
