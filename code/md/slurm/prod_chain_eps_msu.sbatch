#!/bin/bash --login
# One state point of the eps_r campaign, end to end, like prod_chain_msu.sbatch:
# equilibrate, fix the box, zero field, measure the run lengths, write the
# production (p00-p15) and long-wavelength (p16-p21) potentials, submit the
# field array.  At c = 0.04 the mixed/Gaussian families are run a second time
# with new seeds (p22-p29), as the pilot state point had 24 + 6 runs.
#
# The state point lives under $SIP_ROOT/eps${EPS_R}/runs/, NOT runs/: learn/
# picks one directory per concentration from runs/*, and would otherwise mix
# the two dielectric constants.  The force field is a copy of templates/ff.lmp
# with the `dielectric` line replaced, kept in the state directory.
#
#   sbatch -p scavenger -A scavenger -q scavenger \
#          --export=ALL,CONC=0.06,SEED=20260926,EPS_R=2 slurm/prod_chain_eps_msu.sbatch
# c = 0.08 runs on two nodes (256 ranks: 604 vs 458 steps/s, measured):
#   add -N 2 -n 256 --ntasks-per-node=128 and FIELD_NTASKS=256,FIELD_NNODES=2
#SBATCH --job-name=sip_epsch
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --open-mode=append
#SBATCH --requeue
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
source $SIP_ROOT/slurm/_check_ranks.sh
: ${CONC:?}; : ${SEED:?}; : ${EPS_R:?}; : ${TEMP:=1.0}; : ${PPPM:=1.0e-4}
: ${NPUSH:=60000}; : ${NCAP:=20000}; : ${NNVT:=400000}
: ${NNPT:=2000000}           # 10 000 tau of NPT
: ${NZEQ:=400000}            # 2 000 tau of NVT at the fixed box
# zero-field production: 2e4 tau, 6e4 tau for c >= 0.08 (as the eps_r = 7.5 campaign)
if [ -z "$NZF" ]; then
  NZF=$($SIP_PY -c "print(12000000 if $CONC >= 0.08 else 4000000)")
fi
: ${FIELD_PARTITION:=scavenger}; : ${FIELD_NTASKS:=128}; : ${FIELD_NNODES:=1}
NP=${SLURM_NTASKS:-128}
S=$SIP_ROOT/eps${EPS_R}/runs/prod_T${TEMP}_c${CONC}
E=$S/equil; Z=$S/zero_field
mkdir -p $E $Z
FF=$S/ff.lmp; TAB=$S/solvation.table
if [ ! -f $FF ]; then
  sed "s/^dielectric .*/dielectric      $EPS_R/" $SIP_ROOT/templates/ff.lmp > $FF
  cp $SIP_ROOT/tools/solvation.table $TAB
  for f in in.01_pushoff in.02_warmup in.03_npt in.04_zerofield in.05_field; do
    cp $SIP_ROOT/templates/$f $S/; done
fi
grep -q "^dielectric      $EPS_R\$" $FF || { echo "FATAL: $FF dielectric is not $EPS_R"; exit 1; }
echo "### state point c=$CONC eps_r=$EPS_R T=$TEMP seed=$SEED on $NP ranks, $(hostname), $(date)"

# --- 1. build, push off, warm up, NPT at p = 0 -------------------------------
if [ ! -f $E/restart.npt ]; then
  cd $E
  [ -f melt.data ] || $SIP_PY $SIP_ROOT/tools/build_system.py --conc $CONC --rho 0.85 \
       --seed $SEED --out melt.data
  if [ ! -f restart.pushoff ]; then
    srun -n $NP $LMP -log log.01_pushoff -in $S/in.01_pushoff \
         -var data melt.data -var out restart.pushoff -var nsteps $NPUSH \
         -var T $TEMP -var seed $SEED
    check_ranks log.01_pushoff $NP
  fi
  if [ ! -f restart.warmup ]; then
    srun -n $NP $LMP -log log.02_warmup -in $S/in.02_warmup \
         -var restart restart.pushoff -var ff $FF -var tabfile $TAB \
         -var out restart.warmup -var T $TEMP -var seed $((SEED+1)) \
         -var ncap $NCAP -var nnvt $NNVT -var pppm_acc $PPPM
    check_ranks log.02_warmup $NP
  fi
  srun -n $NP $LMP -log log.03_npt -in $S/in.03_npt \
       -var restart restart.warmup -var ff $FF -var tabfile $TAB \
       -var out restart.npt -var T $TEMP -var pdamp 1.0 -var nsteps $NNPT \
       -var pppm_acc $PPPM
  check_ranks log.03_npt $NP
  echo "### NPT done $(date)"
fi

# --- 2. the box for every run of this state point ----------------------------
LT=$($SIP_PY -c "
import numpy as np
d = np.loadtxt('$E/vol.dat', comments='#')
L = d[int(0.3*len(d)):, 1]
print(f'{L.mean():.6f}')")
echo "### <L> = $LT  (V = $($SIP_PY -c "print(f'{$LT**3:.1f}')"))"

# --- 3. re-equilibrate at that box, then zero-field production ---------------
if [ ! -f $Z/restart.zf ]; then
  cd $Z
  srun -n $NP $LMP -log log.04_zerofield -in $S/in.04_zerofield \
       -var restart $E/restart.npt -var ff $FF -var tabfile $TAB -var out restart.zf \
       -var T $TEMP -var Lt $LT -var nequil $NZEQ -var nsteps $NZF -var pppm_acc $PPPM
  check_ranks log.04_zerofield $NP
  echo "### zero field done $(date)"
fi

# --- 4. what the field stage needs (l_B in Sk.npz must carry eps_r) ----------
$SIP_PY $SIP_ROOT/tools/zero_field_report.py --run $Z --conc $CONC --temp $TEMP --eps-r $EPS_R \
        | tee $S/zero_field_report.txt
$SIP_PY -c "
import numpy as np, sys; sys.path.insert(0, '$SIP_ROOT/tools'); import model as M
lb = float(np.load('$Z/Sk.npz')['lB']); ref = M.bjerrum_sigma($TEMP, $EPS_R)
assert abs(lb - ref) < 1e-9, (lb, ref); print(f'### l_B in Sk.npz = {lb:.4f} (eps_r = $EPS_R)')"
read NTRANS NSTEPS < <($SIP_PY -c "
import json; d = json.load(open('$Z/tau_int.json')); print(int(d['ntrans']), int(d['nprod']))")
echo "### field run lengths: transient $NTRANS, production $NSTEPS steps"

# --- 5. potentials at this box, and the runs ---------------------------------
mkdir -p $S/potentials
if [ ! -f $S/potentials/p15.lmp ]; then
  $SIP_PY $SIP_ROOT/tools/make_schedule.py --lz $LT --temp $TEMP --mode production \
          --outdir $S/potentials > $S/potentials/gen.log 2>&1
  $SIP_PY $SIP_ROOT/tools/make_schedule.py --lz $LT --temp $TEMP --mode long \
          --tag-offset 16 --outdir $S/potentials >> $S/potentials/gen.log 2>&1
  if [ "$CONC" = "0.04" ]; then
    $SIP_PY $SIP_ROOT/tools/make_schedule.py --lz $LT --temp $TEMP --mode production \
            --subset 8:16 --tag-offset 22 --outdir $S/potentials >> $S/potentials/gen.log 2>&1
  fi
fi
NPOT=$(ls $S/potentials/p*.lmp | wc -l)
echo "### $NPOT potentials"

cat > $S/submit_field.sh <<EOS
#!/bin/bash
# written by the chain job; rerun by hand if the in-job submission was refused
cd $SIP_ROOT
FF=$FF TAB=$TAB ARRAY=0-$((NPOT-1)) PPPM=$PPPM NTASKS=$FIELD_NTASKS NNODES=$FIELD_NNODES MEMCPU=1G NRST=250000 \\
  PARTITION=$FIELD_PARTITION ACCOUNT=scavenger QOS=scavenger \\
  ./run_state.sh field $S $NTRANS $NSTEPS
EOS
chmod +x $S/submit_field.sh
[ -f $S/field_submitted ] || { bash $S/submit_field.sh && touch $S/field_submitted; } \
  || echo "### could not submit from the compute node; run $S/submit_field.sh"
echo "### chain for c=$CONC eps_r=$EPS_R done $(date)"
set +e; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State; exit 0
