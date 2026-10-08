#!/bin/bash
# Submit the profile analysis of each run as it finishes (2026-10-04): (C) p60/p61 and (D) p62/p63 in the 2x boxes,
# aperiodic 2D p54/p55.  Detached: nohup setsid bash bigbox/watch_analyses_1004.sh > bigbox/watch_1004.log 2>&1 &
S=/mnt/gs21/scratch/lyuliyao/salt_in_polymer; cd $S
RUNS="bigbox/eps75_c0.04_x2:60:18377137 bigbox/eps2_c0.04_x2:60:18377138 bigbox/eps2_c0.04_x2:61:18377138
      bigbox/eps75_c0.04_x2:62:18409557 bigbox/eps75_c0.04_x2:63:18409557 bigbox/eps2_c0.04_x2:62:18409558 bigbox/eps2_c0.04_x2:63:18409558
      field2d/eps75_c0.04:54:18409645 field2d/eps75_c0.04:55:18409645 field2d/eps2_c0.04:54:18409646 field2d/eps2_c0.04:55:18409646"
end=$((SECONDS + 5 * 86400))
while [ $SECONDS -lt $end ]; do
  left=0
  for r in $RUNS; do
    IFS=: read st tag job <<< "$r"; d=$st/field_p$tag; m=$d/.analysis_submitted
    [ -f $m ] && continue; left=$((left + 1))
    l=$(ls $d/log.05_field.* 2>/dev/null | sort -V | tail -1); [ -n "$l" ] || continue
    grep -q "Total wall time" $l || continue
    [ -z "$(squeue -h -j ${job}_$tag 2>/dev/null)" ] || continue
    case $st in
      field2d/*) j=$(sbatch --parsable --export=ALL,STATEDIR=$S/$st --array=$tag slurm/analyze_field2d.sbatch);;
      *)         j=$(sbatch --parsable --export=ALL,STATEDIR=$S/$st --array=$tag --mem=64G --time=08:00:00 --job-name=sip_fana --output=$S/bigbox/field_analysis/sip_fana-$(basename $st)-%A_%a.out slurm/analyze_field_msu.sbatch);;
    esac
    echo "$(date '+%m-%d %H:%M') $st p$tag finished -> analysis $j" ; echo $j > $m
  done
  [ $left = 0 ] && { echo "$(date '+%m-%d %H:%M') all analyses submitted"; break; }
  sleep 900
done
