#!/bin/bash --login
#SBATCH --job-name=aa_diag
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/test_small/%x-%j.out
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
module purge 2>/dev/null
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/test_small
srun -n 16 lmp -in in.eq -log log.eq > screen.eq 2>&1 || exit 1
srun -n 5 --exact lmp -in in.compress -var kmode pppm -var def 0 -var tag A -log log.A > screen.A 2>&1 &
srun -n 5 --exact lmp -in in.compress -var kmode pppm -var def 1 -var tag B -log log.B > screen.B 2>&1 &
srun -n 5 --exact lmp -in in.compress -var kmode dsf  -var def 1 -var tag C -log log.C > screen.C 2>&1 &
wait
for t in A B C; do echo "== $t"; grep -E "ERROR|Loop time" screen.$t | head -2; done
