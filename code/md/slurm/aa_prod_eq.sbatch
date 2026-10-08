#!/bin/bash --login
# Production box r = 0.04: 40 distinct PEO100 chains + 160 LiTFSI, 1.135 g/cm3.
#   build.sh     chains, random placement, fftool, ions x 0.8   -> data.dense.lmp
#   in.soft      slow soft push-off at melt density,
#                purely repulsive, charges off            -> data.soft.lmp
#   in.eq        full force field at melt density: minimise, limited MD, NVT
#                                                              -> data.eq.lmp
#   in.npt       2 ns NPT 400 K, PPPM 1e-5, restarts           -> data.npt.lmp
# Chain Rg is written per stage to rg.<stage>.dat (tools/rg_series.py).
# Each stage is skipped when its output exists and NPT resumes from the newest
# restart.npt.*, so a preempted scavenger job simply requeues and carries on.
#SBATCH --job-name=aa_prod_eq
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/prod_r004/%x-%j.out
#SBATCH --open-mode=append
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/prod_r004
PY=/mnt/home/lyuliyao/.conda/envs/aamd/bin/python
N=$SLURM_NTASKS
echo "### $(hostname) $(date) restart count ${SLURM_RESTART_COUNT:-0}"
if [ ! -s data.dense.lmp ]; then
  ./build.sh
fi
module purge 2>/dev/null
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos
if [ ! -s data.soft.lmp ]; then
  srun -n $N lmp -in in.soft -log log.soft > screen.soft 2>&1
  echo "### soft done $(date)"
fi
if [ ! -s data.eq.lmp ]; then
  srun -n $N lmp -in in.eq -log log.eq > screen.eq 2>&1
  echo "### eq done $(date)"
fi
if [ ! -s data.npt.lmp ]; then
  R=$(ls -t restart.npt.* 2>/dev/null | head -1)
  srun -n $N lmp -in in.npt $([ -z "$R" ] && echo "-var fresh 1" || echo "-var fresh 0 -var rst $R") -log log.npt.$(date +%s) > screen.npt.$(date +%s) 2>&1
  echo "### npt done $(date)"
fi
