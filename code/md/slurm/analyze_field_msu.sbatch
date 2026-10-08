#!/bin/bash --login
# Section 7 acceptance for the field runs: YBG (bin by bin and mode by mode),
# the first/second-half profile comparison, and the mean ion current.
# One array task per run; the dumps are ~3.4 GB per run, so they get their own core.
#SBATCH --job-name=sip_fana
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/field_analysis/%x-%A_%a.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
: ${STATEDIR:?}; : ${POTDIR:=$STATEDIR/potentials}
TAG=$(printf "p%02d" ${SLURM_ARRAY_TASK_ID:-0})
RUN=$STATEDIR/field_$TAG
test -d $RUN || { echo "no $RUN"; exit 1; }
echo "### acceptance for $RUN  $(date)"
# lz and the highest driven mode come from the potential itself: the box differs
# between state points, and the mixed/gauss columns drive modes far above the
# old defaults (mmax 10 and 7).  Modes whose amplitude is below 1e-3 of the
# largest are Fourier tail, not signal, so they do not set the cutoff.
read LZ MMAX < <($SIP_PY - "$POTDIR/$TAG.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
terms = d.get("neutral", []) + d.get("charged", [])
amax = max([t["A"] for t in terms], default=1.0)
ms = [t["m"] for t in terms if t["A"] >= 1e-3 * amax] or [6]
print(f"{d['lz']:.6f} {max(ms) + 2}")
EOF
)
echo "### lz = $LZ, mmax = $MMAX"
$SIP_PY $SIP_ROOT/tools/analyze_profiles.py --run $RUN --pot $POTDIR/$TAG.json \
        --mmax $MMAX --out $RUN/profiles.npz
$SIP_PY $SIP_ROOT/tools/field_acceptance.py --run $RUN --pot $POTDIR/$TAG.json \
        --out $RUN/acceptance.json
echo "### acceptance $TAG done $(date)"
