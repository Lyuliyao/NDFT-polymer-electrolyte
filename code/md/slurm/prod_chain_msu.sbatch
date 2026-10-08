#!/bin/bash --login
# One state point, end to end: equilibrate, fix the box, run zero field,
# measure what the field stage needs, and submit the 16 field runs.
#
#   sbatch --export=ALL,CONC=0.06,SEED=...,PPPM=1.0e-4 slurm/prod_chain_msu.sbatch
#
# Each stage is skipped when its output is already there, so a preemption costs
# at most the stage in flight rather than the whole chain.
#SBATCH --job-name=sip_chain
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
: ${CONC:?}; : ${TEMP:=1.0}; : ${PPPM:=1.0e-4}
: ${SEED:=20260923}; : ${NPUSH:=60000}; : ${NCAP:=20000}; : ${NNVT:=400000}
: ${NNPT:=2000000}           # 10 000 tau of NPT
: ${NZEQ:=400000}            # 2 000 tau of NVT at the fixed box
: ${NZF:=4000000}            # 20 000 tau of zero-field production
: ${FIELD_PARTITION:=scavenger}; : ${FIELD_NTASKS:=32}
NP=${SLURM_NTASKS:-32}
S=$SIP_ROOT/runs/prod_T${TEMP}_c${CONC}
E=$S/equil; Z=$S/zero_field
FF=$SIP_ROOT/templates/ff.lmp
TAB=$SIP_ROOT/tools/solvation.table
mkdir -p $E $Z
echo "### state point c=$CONC T=$TEMP pppm=$PPPM on $NP ranks, $(hostname), $(date)"

# --- 1. build, push off, warm up, NPT at p = 0 -------------------------------
if [ ! -f $E/restart.npt ]; then
  cd $E
  [ -f melt.data ] || $SIP_PY $SIP_ROOT/tools/build_system.py --conc $CONC --rho 0.85 \
       --seed $SEED --out melt.data
  if [ ! -f restart.pushoff ]; then
    srun -n $NP $LMP -log log.01_pushoff -in $SIP_ROOT/templates/in.01_pushoff \
         -var data melt.data -var out restart.pushoff -var nsteps $NPUSH \
         -var T $TEMP -var seed $SEED
    check_ranks log.01_pushoff $NP
  fi
  if [ ! -f restart.warmup ]; then
    srun -n $NP $LMP -log log.02_warmup -in $SIP_ROOT/templates/in.02_warmup \
         -var restart restart.pushoff -var ff $FF -var tabfile $TAB \
         -var out restart.warmup -var T $TEMP -var seed $((SEED+1)) \
         -var ncap $NCAP -var nnvt $NNVT -var pppm_acc $PPPM
    check_ranks log.02_warmup $NP
  fi
  srun -n $NP $LMP -log log.03_npt -in $SIP_ROOT/templates/in.03_npt \
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
  srun -n $NP $LMP -log log.04_zerofield -in $SIP_ROOT/templates/in.04_zerofield \
       -var restart $E/restart.npt -var ff $FF -var tabfile $TAB -var out restart.zf \
       -var T $TEMP -var Lt $LT -var nequil $NZEQ -var nsteps $NZF -var pppm_acc $PPPM
  check_ranks log.04_zerofield $NP
  echo "### zero field done $(date)"
fi

# --- 4. what the field stage needs ------------------------------------------
$SIP_PY $SIP_ROOT/tools/zero_field_report.py --run $Z --conc $CONC --temp $TEMP \
        | tee $S/zero_field_report.txt
read NTRANS NSTEPS < <($SIP_PY -c "
import json; d = json.load(open('$Z/tau_int.json')); print(int(d['ntrans']), int(d['nprod']))")
echo "### field run lengths: transient $NTRANS, production $NSTEPS steps"

# --- 5. the 16 potentials at this box, and the runs --------------------------
mkdir -p $S/potentials      # the redirect below needs it to exist already
$SIP_PY $SIP_ROOT/tools/make_schedule.py --lz $LT --temp $TEMP --mode production \
        --outdir $S/potentials > $S/potentials/gen.log 2>&1
tail -1 $S/potentials/gen.log

cat > $S/submit_field.sh <<EOF
#!/bin/bash
# written by the chain job; rerun by hand if the in-job submission was refused
cd $SIP_ROOT
ARRAY=0-15%16 PPPM=$PPPM NTASKS=$FIELD_NTASKS NRST=250000 \\
  PARTITION=$FIELD_PARTITION ACCOUNT=scavenger QOS=scavenger \\
  ./run_state.sh field $S $NTRANS $NSTEPS
EOF
chmod +x $S/submit_field.sh
bash $S/submit_field.sh || echo "### could not submit from the compute node; run $S/submit_field.sh"
echo "### chain for c=$CONC done $(date)"
set +e; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State; exit 0
