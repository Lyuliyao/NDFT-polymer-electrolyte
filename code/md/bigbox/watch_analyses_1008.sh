#!/bin/bash
# Submit the profile analysis of each run as it finishes (2026-10-08): (C) p60/p61 and (D) p62/p63 in the 4x boxes (x4 analyses need 128 GB).
# aperiodic 2D p54/p55.  Detached: nohup setsid bash bigbox/watch_analyses_1004.sh > bigbox/watch_1004.log 2>&1 &
S=/mnt/gs21/scratch/lyuliyao/salt_in_polymer; cd $S
RUNS="bigbox/eps75_c0.04_x4:60:18680004 bigbox/eps75_c0.04_x4:61:18680004 bigbox/eps75_c0.04_x4:62:18680005 bigbox/eps75_c0.04_x4:63:18680005 bigbox/eps2_c0.04_x4:60:18680006 bigbox/eps2_c0.04_x4:61:18680006 bigbox/eps2_c0.04_x4:62:18680007 bigbox/eps2_c0.04_x4:63:18680007"
end=$((SECONDS + 30 * 86400))
while [ $SECONDS -lt $end ]; do
  left=0
  for r in $RUNS; do
    IFS=: read st tag job <<< "$r"; d=$st/field_p$tag; m=$d/.analysis_submitted
    [ -f $m ] && continue; left=$((left + 1))
    l=$(ls $d/log.05_field.* 2>/dev/null | sort -V | tail -1); [ -n "$l" ] || continue
    grep -q "Total wall time" $l || continue
    [ -z "$(squeue -h -j ${job}_$tag 2>/dev/null)" ] || continue
    case $st in
      *)         j=$(sbatch --parsable --export=ALL,STATEDIR=$S/$st --array=$tag --mem=128G --time=12:00:00 --job-name=sip_fana --output=$S/bigbox/field_analysis/sip_fana-$(basename $st)-%A_%a.out slurm/analyze_field_msu.sbatch);;
    esac
    echo "$(date '+%m-%d %H:%M') $st p$tag finished -> analysis $j" ; echo $j > $m
  done
  [ $left = 0 ] && { echo "$(date '+%m-%d %H:%M') all analyses submitted"; break; }
  sleep 900
done
