#!/bin/bash
# Driver for one state point.  Each stage depends on a number measured by the
# previous one, so the stages are deliberately separate commands.
#
#   ./run_state.sh equil     <state> <conc> [temp] [seed]
#   ./run_state.sh analyze-npt <state>            -> mean L, density, drift
#   ./run_state.sh zerofield <state> [nsteps]     -> needs the mean L
#   ./run_state.sh analyze-zf  <state>            -> D_s, sigma/sigma_NE, S_ZZ(k)
#   ./run_state.sh potentials  <state> [pilot|production]
#   ./run_state.sh field       <state> [ntrans] [nsteps]
#   ./run_state.sh analyze-field <state>
set -e
usage() { sed -n '4,11p' "$0"; exit 1; }
[ $# -ge 2 ] || usage
source "$(dirname "$0")/env.sh"
# batch scripts per site: slurm/<stage>.sbatch (Bridges-2), slurm/<stage>_msu.sbatch (MSU)
SB=""; [ "$SIP_SITE" = msu ] && SB=_msu
CMD=$1; STATE=$(readlink -f "$2"); shift 2

case $CMD in
equil)
  CONC=$1; TEMP=${2:-1.0}; SEED=${3:-20260920}
  mkdir -p "$STATE"
  sbatch -n 32 -t 8:00:00 -o "$STATE/equil.%j.out" \
    --export=ALL,RUNDIR=$STATE/equil,CONC=$CONC,TEMP=$TEMP,SEED=$SEED \
    $SIP_ROOT/slurm/equil$SB.sbatch
  ;;

analyze-npt)
  $SIP_PY $SIP_ROOT/tools/analyze_zerofield.py --npt "$STATE/equil" \
     --run "$STATE/equil" --out "$STATE/npt.json" 2>/dev/null \
  || $SIP_PY - "$STATE" <<'PY'
import sys, os, json, numpy as np
sys.path.insert(0, os.environ["SIP_ROOT"] + "/tools")
import dumpio
st = sys.argv[1]
_, v = dumpio.read_ave_time(os.path.join(st, "equil", "vol.dat"))
n0 = int(0.3 * len(v)); L = v[n0:, 1]
nb = 10; Lb = np.array([L[i::nb].mean() for i in range(nb)])
res = dict(L_mean=float(L.mean()), L_err=float(Lb.std(ddof=1)/np.sqrt(nb)),
           rho_all=float(v[n0:,2].mean()), rho_mono=float(v[n0:,3].mean()),
           press=float(v[n0:,4].mean()),
           drift=float(np.polyfit(np.arange(len(L)), L, 1)[0]*len(L)))
json.dump(res, open(os.path.join(st, "npt.json"), "w"), indent=2)
print(f"L = {res['L_mean']:.4f} +- {res['L_err']:.4f} sigma   "
      f"rho_all = {res['rho_all']:.5f}   rho_mono = {res['rho_mono']:.5f}")
print(f"<P> = {res['press']:+.5f} (target 0)   drift over window = {res['drift']:+.4f}")
PY
  ;;

zerofield)
  NSTEPS=${1:-4000000}
  LT=$($SIP_PY -c "import json;print(f\"{json.load(open('$STATE/npt.json'))['L_mean']:.6f}\")")
  echo "zero-field run at L = $LT sigma, $NSTEPS steps"
  sbatch -n 32 -t 24:00:00 -o "$STATE/zf.%j.out" \
    --export=ALL,RUNDIR=$STATE/zerofield,SRCRESTART=$STATE/equil/restart.npt,LT=$LT,NSTEPS=$NSTEPS \
    $SIP_ROOT/slurm/zerofield$SB.sbatch
  ;;

analyze-zf)
  CONC=$(grep -m1 "c_LJ=" "$STATE/equil/melt.data" | sed 's/.*c_LJ=\([0-9.]*\).*/\1/')
  $SIP_PY $SIP_ROOT/tools/analyze_zerofield.py --npt "$STATE/equil" \
     --run "$STATE/zerofield" --conc ${CONC:-0.04} \
     --out "$STATE/zerofield.json" "$@"
  $SIP_PY $SIP_ROOT/tools/plot_report.py --state "$STATE" --json "$STATE/zerofield.json"
  ;;

potentials)
  MODE=${1:-production}
  LT=$($SIP_PY -c "import json;print(f\"{json.load(open('$STATE/npt.json'))['L_mean']:.6f}\")")
  $SIP_PY $SIP_ROOT/tools/make_schedule.py --lz $LT --mode $MODE \
     --outdir "$STATE/potentials"
  ;;

field)
  # Default run lengths come from the measured D_s via tau_lambda = lambda^2/(4 pi^2 D_s)
  # with lambda = 8 sigma, the longest wavelength in the potential:
  # transient 10 tau_lambda, production 100 tau_lambda (Section 6 of the plan).
  if [ -z "$1" ] && [ -f "$STATE/zerofield.json" ]; then
    read NTRANS NSTEPS < <($SIP_PY -c "
import json,math
t=json.load(open('$STATE/zerofield.json'))['tau_lambda_8']
r=lambda x:int(round(x/2e5)*2e5)
print(r(10*t/0.005), r(100*t/0.005))")
    echo "run lengths from measured D_s: transient $NTRANS, production $NSTEPS steps"
  fi
  NTRANS=${NTRANS:-${1:-1100000}}; NSTEPS=${NSTEPS:-${2:-11000000}}
  N=$(ls "$STATE"/potentials/p*.lmp | wc -l)
  # the zero-field restart, or the short re-run that replaced it
  # SRCRESTART names the start explicitly (not checked: with DEP it may not exist yet)
  if [ -n "$SRCRESTART" ]; then SRC=$SRCRESTART; else
  SRC=$STATE/zero_field/restart.zf
  [ -f "$SRC" ] || SRC=$STATE/zerofield/restart.zf
  [ -f "$SRC" ] || SRC=$STATE/zerofield_tail/restart.zf
  [ -f "$SRC" ] || { echo "no zero-field restart under $STATE"; exit 1; }
  fi
  # ARRAY=0-1 runs the pilot pair only; PPPM overrides the k-space accuracy
  ARRAY=${ARRAY:-0-$((N-1))%8}
  echo "submitting field runs $ARRAY of $N, transient $NTRANS, production $NSTEPS steps"
  echo "  restart $SRC, pppm ${PPPM:-1.0e-4}, ${NTASKS:-32} ranks"
  # PARTITION=scavenger uses the preemptible pool (it needs ACCOUNT=scavenger
  # QOS=scavenger too); the runs checkpoint every NRST steps and the batch
  # script resumes from the newest checkpoint.  The amd20 constraint in the
  # batch script must stay: the binary is built with -march=znver2 and would
  # die with an illegal instruction on the Intel nodes scavenger also holds.
  PART=${PARTITION:+-p $PARTITION}${ACCOUNT:+ -A $ACCOUNT}${QOS:+ -q $QOS}
  # MEMCPU overrides the batch script's --mem-per-cpu (1G lets 128 ranks fit a 256 GB node);
  # NNODES spreads NTASKS over that many whole nodes (e.g. 256 ranks on 2); TLIM the wall time;
  # NDUMP/NREP (dump interval, frames per profile block) pass through to the batch script
  sbatch -n ${NTASKS:-32} -t ${TLIM:-48:00:00} --array=$ARRAY -o "$STATE/field.%A_%a.out" $PART ${DEP:+--dependency=$DEP} ${CONSTRAINT:+--constraint=$CONSTRAINT} ${MEMCPU:+--mem-per-cpu=$MEMCPU} ${NNODES:+-N $NNODES --ntasks-per-node=$((NTASKS/NNODES))} \
    --export=ALL,STATEDIR=$STATE,SRCRESTART=$SRC,NTRANS=$NTRANS,NSTEPS=$NSTEPS,PPPM=${PPPM:-1.0e-4},NRST=${NRST:-500000} \
    $SIP_ROOT/slurm/field$SB.sbatch
  ;;

analyze-field)
  for d in "$STATE"/field_p*; do
    tag=$(basename $d | sed 's/field_//')
    [ -f "$d/cat.dump" ] || [ -f "$d/cat.dump.gz" ] || continue
    $SIP_PY $SIP_ROOT/tools/analyze_profiles.py --run "$d" \
       --pot "$STATE/potentials/$tag.json" "$@"
  done
  ;;

*) usage ;;
esac
