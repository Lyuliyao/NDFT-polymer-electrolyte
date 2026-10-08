#!/bin/bash --login
# All-atom zero-field production, r = 0.04 (aamd/prod_r004/zf/in.prod), 500 ns.
# Each segment resumes from the newest *complete* restart.prod.* (truncated files,
# e.g. from a full filesystem, are skipped by size), stops itself before the
# walltime and resubmits until data.prod.lmp exists.  An older restart is tried
# only if LAMMPS fails before the first thermo line; a crash mid-run just
# resubmits (at most 3 consecutive failures, counted in fail.count).
#SBATCH --job-name=aa_zf_r004
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/prod_r004/zf/%x-%j.out
#SBATCH --open-mode=append
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
module purge 2>/dev/null
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos
SELF=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/slurm/aa_zf.sbatch
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/prod_r004/zf
echo "### $(hostname) $(date) job $SLURM_JOB_ID restart count ${SLURM_RESTART_COUNT:-0}"
[ -s data.prod.lmp ] && { echo "already done"; exit 0; }

# complete restarts = the largest size present; newest first
full=$(ls -l restart.prod.* 2>/dev/null | awk '{print $5}' | sort -n | tail -1)
CAND=$(for f in restart.prod.*; do [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$full" ] && echo "$f"; done | sort -t. -k3 -n -r)
# prune: keep the three newest complete restarts, drop incomplete ones
echo "$CAND" | tail -n +4 | xargs -r rm -f
for f in restart.prod.*; do [ -f "$f" ] && [ "$(stat -c %s "$f")" != "$full" ] && { echo "### removing incomplete $f"; rm -f "$f"; }; done
CAND=$(echo "$CAND" | head -2)

ok=0; started=0
for R in ${CAND:-none}; do
  echo "### start from $R"
  T=$(date +%s)
  if [ "$R" = none ]; then ARGS="-var fresh 1"; else ARGS="-var fresh 0 -var rst $R"; fi
  srun -n $SLURM_NTASKS lmp -in in.prod $ARGS -log log.$T > screen.$T 2>&1 && { ok=1; break; }
  if grep -qE "^ +Step " log.$T; then started=1; echo "### lmp failed mid-run from $R"; break; fi
  echo "### lmp failed before running from $R"
done

if [ $ok = 1 ]; then
  rm -f fail.count
elif [ $started = 1 ]; then
  n=$(( $(cat fail.count 2>/dev/null || echo 0) + 1 )); echo $n > fail.count
  [ $n -ge 3 ] && { echo "### $n consecutive mid-run failures, stopping"; exit 1; }
else
  echo "### giving up"; exit 1
fi
if [ ! -s data.prod.lmp ]; then
  echo "### segment ended $(date), resubmitting"
  sbatch $SELF
fi
