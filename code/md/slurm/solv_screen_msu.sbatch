#!/bin/bash --login
# Zero-field state point at a non-production solvation strength S:
# table for S -> equil (build, push-off, warm-up, NPT) -> mean L -> zero field
# -> D, pairing.  One job, so the NPT volume feeds the NVT run without a
# manual step.  Table and force field are copied into the state directory.
# EPS_R (default 7.5) replaces the `dielectric` line of the copied force field.
#   sbatch --export=ALL,STATE=<dir>,SOLV_S=<S>,EPS_R=<eps>,CONC=0.04 slurm/solv_screen_msu.sbatch
#SBATCH --job-name=sip_solvS
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=3:59:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${STATE:?}; : ${SOLV_S:=4.33}; : ${EPS_R:=7.5}; : ${CONC:=0.04}; : ${TEMP:=1.0}; : ${SEED:=20260920}
: ${NPUSH:=60000}; : ${NCAP:=20000}; : ${NNVT:=400000}; : ${NNPT:=2000000}
: ${NEQUIL:=200000}; : ${NSTEPS:=4000000}; : ${NDUMP:=200}; : ${PPPM:=1.0e-4}
NP=${SLURM_NTASKS:-128}
mkdir -p $STATE/equil $STATE/zerofield
cd $STATE
TAB=$STATE/solvation_S$SOLV_S.table; FF=$STATE/ff.lmp
[ -f $TAB ] || $SIP_PY $SIP_ROOT/tools/make_solv_table.py --S $SOLV_S --out $TAB
cp -n $SIP_ROOT/templates/ff.lmp $FF
[ -f $FF.eps ] || { sed -i "s/^dielectric .*/dielectric      $EPS_R/" $FF; echo $EPS_R > $FF.eps; }
grep -q "^dielectric      $EPS_R\$" $FF || { echo "FATAL: $FF dielectric is not $EPS_R"; exit 1; }
for f in in.01_pushoff in.02_warmup in.03_npt in.04_zerofield; do cp -n $SIP_ROOT/templates/$f $STATE/; done
echo "### S=$SOLV_S eps_r=$EPS_R c=$CONC T=$TEMP seed=$SEED on $NP ranks"

cd $STATE/equil
# each stage is skipped when its restart exists, so a resubmission resumes
if [ ! -f restart.warmup ]; then
  $SIP_PY $SIP_ROOT/tools/build_system.py --conc $CONC --rho 0.85 --seed $SEED --out melt.data
  srun -n $NP $LMP -log log.01_pushoff -in $STATE/in.01_pushoff \
       -var data melt.data -var out restart.pushoff -var nsteps $NPUSH -var T $TEMP -var seed $SEED
  check_ranks log.01_pushoff $NP
  srun -n $NP $LMP -log log.02_warmup -in $STATE/in.02_warmup \
       -var restart restart.pushoff -var ff $FF -var tabfile $TAB \
       -var out restart.warmup -var T $TEMP -var seed $((SEED+1)) \
       -var ncap $NCAP -var nnvt $NNVT -var pppm_acc $PPPM
  check_ranks log.02_warmup $NP
fi
if [ ! -f restart.npt ]; then
  srun -n $NP $LMP -log log.03_npt -in $STATE/in.03_npt \
       -var restart restart.warmup -var ff $FF -var tabfile $TAB \
       -var out restart.npt -var T $TEMP -var pdamp 1.0 -var nsteps $NNPT -var pppm_acc $PPPM
  check_ranks log.03_npt $NP
fi
cd $STATE
$SIP_ROOT/run_state.sh analyze-npt $STATE
LT=$($SIP_PY -c "import json;print(f\"{json.load(open('$STATE/npt.json'))['L_mean']:.6f}\")")

cd $STATE/zerofield
srun -n $NP $LMP -log log.04_zerofield -in $STATE/in.04_zerofield \
     -var restart $STATE/equil/restart.npt -var ff $FF -var tabfile $TAB -var out restart.zf \
     -var T $TEMP -var Lt $LT -var nequil $NEQUIL -var nsteps $NSTEPS \
     -var ndump $NDUMP -var pppm_acc $PPPM
check_ranks log.04_zerofield $NP

cd $STATE
$SIP_ROOT/run_state.sh analyze-zf $STATE --eps-r $EPS_R || echo "analyze-zf failed (non-fatal)"
$SIP_PY $SIP_ROOT/tools/ion_pairing.py --dump $STATE/zerofield/ions.dump --stride 5 \
     --out $STATE/pairing.json
$SIP_PY $SIP_ROOT/tools/ion_pairing.py --dump $STATE/zerofield/ions.dump --stride 5 \
     --rc 1.3 --out $STATE/pairing_contact.json
echo "### done"
