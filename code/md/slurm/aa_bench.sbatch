#!/bin/bash --login
# How fast is an all-atom electrolyte of this size on this machine?  Needed to
# turn the (est.) throughput in the all-atom plan into a real number before
# committing to a 7e5 core-hour campaign.
#SBATCH --job-name=aa_bench
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/aa_bench/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
module purge 2>/dev/null
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/aa_bench
echo "### $(hostname) $(date)"; lmp -h 2>/dev/null | head -2 | tail -1
for cfg in "12.0 1e-4" "10.0 1e-5"; do
  set -- $cfg
  for np in 64 128; do
    srun -n $np lmp -log log.$1.$2.$np -in in.bench -var rc $1 -var acc $2 > /dev/null 2>&1 || true
    printf "cutoff %s A, pppm %s, %3d cores: " $1 $2 $np
    grep -E "^Performance" log.$1.$2.$np | tail -1 | awk '{print $2" ns/day,  "$4" timesteps/s"}'
  done
done
echo "### done $(date)"
set +e; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State; exit 0
