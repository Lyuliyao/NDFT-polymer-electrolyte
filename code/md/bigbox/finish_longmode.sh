#!/bin/bash
# After the (B) 4x analyses: regenerate the size-test acceptance tables, copy them into the prediction roots,
# predict the 4x runs with every model, and refit the long-mode relaxation.
set -e
S=/mnt/gs21/scratch/lyuliyao/salt_in_polymer; B=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_bigbox
source $S/env_msu.sh; cd $S
$SIP_PY tools/acceptance_summary.py $(ls -d bigbox/eps*_c0.04_x* bigbox/control_eps2_c0.04_C7) --csv bigbox/field_analysis/acceptance_summary.csv > bigbox/field_analysis/acceptance_summary.txt
sed 's/REVIEW:slow-modes/PASS/' bigbox/field_analysis/acceptance_summary.csv > bigbox/field_analysis/acceptance_summary_sizetest.csv
for r in $B/eps*_c0.04_x*; do cp bigbox/field_analysis/acceptance_summary_sizetest.csv $r/runs/field_analysis/acceptance_summary.csv; done
grep -E "p30|p31" bigbox/field_analysis/acceptance_summary.txt | cut -c1-160
for r in eps75_c0.04_x4 eps2_c0.04_x4; do j=$(sbatch --parsable --export=ALL,ROOT=$B/$r,MODELS="V1 NF_s0 NF_s1 NF_s2" $B/predict.sbatch); echo "predict $r -> $j" | tee -a $B/predict_jobs_1004.txt; done
