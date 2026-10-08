#!/bin/bash --login
# How much does PPPM accuracy cost, and how far does this system scale?
# Short field runs at 1e-4/1e-5 on 32/64/128 ranks, plus single-configuration
# forces at 1e-4/1e-5/1e-6 so the k-space error can be compared with the
# thermal force fluctuation that sets the YBG error bars.
#SBATCH --job-name=sip_bench
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/field_bench/%x-%j.SLURMout
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU   # stale values from a submitting
# environment that is itself inside a SLURM job collide with --mem-per-cpu in srun
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
STATE=$SIP_ROOT/runs/pilot_T1.0_c0.04_one_both
POT=$STATE/potentials/p01.lmp          # the strong potential: largest external forces
RESTART=$STATE/zerofield_tail/restart.zf
FF=$SIP_ROOT/templates/ff.lmp
TAB=$SIP_ROOT/tools/solvation.table
cd $SIP_ROOT/runs/field_bench

echo "### forces at one configuration, three accuracies"
for acc in 1.0e-4 1.0e-5 1.0e-6; do
  srun -n 32 $LMP -log log.force.$acc -in in.forcecheck -var restart $RESTART -var ff $FF \
       -var tabfile $TAB -var potfile $POT -var tag $acc -var pppm_acc $acc > /dev/null
  grep -E "estimated absolute RMS force accuracy|grid = " log.force.$acc | tr '\n' ' '; echo " [$acc]"
done

echo "### scaling: steps/s for the real field input"
for acc in 1.0e-4 1.0e-5; do
  for np in 32 64 128; do
    srun -n $np $LMP -log log.scale.$acc.$np -in $SIP_ROOT/templates/in.05_field \
         -var restart $RESTART -var ff $FF -var tabfile $TAB -var potfile $POT \
         -var out /dev/null -var ntrans 2000 -var nsteps 4000 -var ndump 200 \
         -var pppm_acc $acc > /dev/null
    printf "pppm %s  %3d ranks: " $acc $np
    grep "Performance" log.scale.$acc.$np | tail -1 | awk '{print $4" timesteps/s"}'
    rm -f cat.dump ani.dump prof_cat.dat prof_ani.dat current.dat
  done
done
echo "### bench done"
set +e; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State; exit 0
