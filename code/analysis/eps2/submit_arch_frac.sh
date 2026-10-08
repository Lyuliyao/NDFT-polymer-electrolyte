#!/bin/bash
# Phase 2 of the comparison of architectures: training-set size.  For each eps
# root, the window / stencil of each baseline is chosen by the mean VALIDATION
# chi2 over its three seeds (never by held-out or c = 0.05 scores); ours and the
# two chosen baselines are retrained on nested, stratified fractions of the
# training runs (validation and held-out runs unchanged), three seeds each.
#   bash submit_arch_frac.sh 0.25 0.5
set -e
FRACS="$*"; [ -n "$FRACS" ] || { echo "usage: $0 <fractions>"; exit 1; }
R=/mnt/research/MultiscaleML_group/Liyao
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
T=c001_c002_c004_c006_c008
for E in "eps2 full" "eps75 long"; do
  set -- $E; ROOT=$R/salt_in_polymer_$1; TAG=$2
  pick() {  # pick <variant> <arch arg> <arch tag> ... : echo the arg with the lowest mean val_chi2
    $PY - "$ROOT" "$TAG" "$@" <<'PYEOF'
import json, sys, numpy as np
root, tag, *spec = sys.argv[1:]
best = None
for i in range(0, len(spec), 2):
    arg, at = spec[i], spec[i + 1]
    v = [json.load(open(f"{root}/runs/learn/joint_{at}_L1_C4_R128x128_H64_c001_c002_c004_c006_c008_val3_{tag}_s{s}/metrics.json"))["val_chi2"] for s in (0, 1, 2)]
    print(f"    {at}: validation chi2 {np.mean(v):.3f} +- {np.std(v, ddof=1):.3f}", file=sys.stderr)
    if best is None or np.mean(v) < best[0]:
        best = (np.mean(v), arg)
print(best[1])
PYEOF
  }
  echo "== $1"
  C1=$(pick "--win-bins 30" c1win_W30 "--win-bins 60" c1win_W60)
  CA=$(pick "--cace-qcut 3" cace_a5q3 "--cace-qcut 6" cace_a5q6)
  echo "   chosen: c1win $C1, cace $CA"
  shift 0
  for FRAC in $FRACS; do
    for VA in "v2|--M 8" "c1win|$C1" "cace|$CA"; do
      V=${VA%%|*}; AR=${VA#*|}
      for S in 0 1 2; do
        J=$(ROOT=$ROOT VARIANT=$V ARCH="$AR" SEED=$S TAG=$TAG FRAC=$FRAC sbatch --parsable --export=ALL $R/salt_in_polymer_eps2/learn_arch.sbatch)
        echo "$1 $V $AR frac $FRAC s$S -> $J"
      done
    done
  done
done
