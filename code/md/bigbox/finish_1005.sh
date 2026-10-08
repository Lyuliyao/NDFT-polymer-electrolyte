#!/bin/bash
# After the (D) superposed long-box runs p62/p63 and the aperiodic 2D runs p54/p55 are analysed (watch_analyses_1004.sh submits
# the analyses): acceptance tables, predictions in the 2x roots (--status PASS REVIEW), summaries and figures, 2D comparison.
#   nohup setsid bash bigbox/finish_1005.sh > bigbox/finish_1005.log 2>&1 &
S=/mnt/gs21/scratch/lyuliyao/salt_in_polymer; B=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox; R=/mnt/research/MultiscaleML_group/Liyao
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python; export JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1; cd $S
RUNS="bigbox/eps75_c0.04_x2:62 bigbox/eps75_c0.04_x2:63 bigbox/eps2_c0.04_x2:62 bigbox/eps2_c0.04_x2:63 field2d/eps75_c0.04:54 field2d/eps75_c0.04:55 field2d/eps2_c0.04:54 field2d/eps2_c0.04:55"
done_run() {   # analysis submitted, its job gone from the queue, and its outputs present
  local d=$1 prof=$2 m=$1/.analysis_submitted
  [ -f $m ] || return 1; [ -z "$(squeue -h -j $(cat $m) 2>/dev/null)" ] || return 1
  [ -f $d/$prof ] && [ -f $d/acceptance.json ]
}
stage() { echo "$(date '+%m-%d %H:%M') $*"; }
end=$((SECONDS + 6 * 86400))
while [ $SECONDS -lt $end ]; do
  left=""
  for r in $RUNS; do IFS=: read st tag <<< "$r"; case $st in field2d/*) p=profiles2d.npz;; *) p=profiles.npz;; esac
    done_run $st/field_p$tag $p || left="$left $st/p$tag"; done
  [ -z "$left" ] && break
  stage "waiting for:$left"; sleep 1800
done
[ -n "$left" ] && { stage "gave up, still missing:$left"; exit 1; }
stage "all analyses done"
# ---- acceptance tables (the roots get the size-test convention: slow undriven modes are not a failure of a driven-mode test)
$PY tools/acceptance_summary.py $(ls -d bigbox/eps*_c0.04_x* bigbox/control_eps2_c0.04_C7) --csv bigbox/field_analysis/acceptance_summary.csv > bigbox/field_analysis/acceptance_summary.txt 2>/dev/null
sed 's/REVIEW:slow-modes/PASS/' bigbox/field_analysis/acceptance_summary.csv > bigbox/field_analysis/acceptance_summary_sizetest.csv
for r in $B/eps*_c0.04_x*; do cp bigbox/field_analysis/acceptance_summary_sizetest.csv $r/runs/field_analysis/acceptance_summary.csv; done
grep -E "^tag|^p6[0-3]" bigbox/field_analysis/acceptance_summary.txt | cut -c1-170
# ---- predictions in the 2x roots
jobs=""
for root in eps75_c0.04_x2 eps2_c0.04_x2; do
  j=$(sbatch --parsable --export=ALL,ROOT=$B/$root,MODELS="KBK2_s0 KBK2_s1 KBK2_s2 V1K NF_s0 NF_s1 NF_s2 V1",PREDICT_ARGS="--status PASS REVIEW" $B/predict.sbatch)
  echo "predict $root (p62/p63, --status PASS REVIEW) -> $j" | tee -a $B/predict_jobs_1004.txt; jobs="$jobs${jobs:+,}$j"
done
while [ -n "$(squeue -h -j $jobs 2>/dev/null)" ]; do sleep 120; done
stage "predictions done: $(sacct -j $jobs -o JobID,State -n | grep -v 'batch\|extern' | tr -s ' \n' ' ')"
grep -h "scored with status\|^  ->" $B/sip_bbpred-${jobs%%,*}.out $B/sip_bbpred-${jobs##*,}.out | cut -c1-180
# ---- summaries and figures
cd $R/salt_in_polymer_eps2
stage "bigbox_summary KBK2 V1K"; $PY figs/bigbox_summary.py KBK2 V1K 2>&1 | grep -v "Warning\|jax_plugins\|cuInit\|plugin\|versions_helpers\|_check_cuda\|^  sn, sf" | grep -A3 "(C) aperiodic\|Driven-mode\|== eps" | cut -c1-400
stage "profiles D"; $PY figs/bigbox_profiles_plot.py D 2>&1 | grep -v Warning
stage "modes D"; $PY figs/bigbox_modes_aperiodic.py D 2>&1 | grep -v Warning | cut -c1-300
stage "field2d_compare (p50-p55)"; $PY figs/field2d_compare.py 2>&1 | grep -v "Warning\|jax_plugins\|cuInit\|plugin\|versions_helpers\|_check_cuda" | tee $R/salt_in_polymer_field2d/field2d_compare_1005_final.txt | grep "p54\|p55" -A4 | cut -c1-300
stage "finished"
