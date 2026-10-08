#!/bin/bash --login
# All-atom toolchain test on the 8-chain PEO20/LiTFSI box.
#   in.eq        minimise, NVT, NVE (SHAKE + 2 fs check)     -> data.eq.lmp
#   in.compress  fix deform to 1.0 g/cm3 under coul/dsf        -> data.compress.lmp
#   in.npt       NPT 400 K with PPPM                           -> data.npt.lmp
#   in.sitefield ion-site external potential, NVE check, per-molecule force check
# STAGE=npt skips the first two, STAGE=site the first three.
#SBATCH --job-name=aa_test
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/test_small/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
module purge 2>/dev/null
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer/aamd/test_small
echo "### $(hostname) $(date)"
if [ "${STAGE:-all}" = all ]; then
  srun -n 16 lmp -in in.eq -log log.eq > screen.eq 2>&1
  srun -n 16 lmp -in in.compress -var kmode dsf -var def 1 -var tag prod -log log.compress > screen.compress 2>&1
  mv data.compress.prod.lmp data.compress.lmp
fi
[ "${STAGE:-all}" = site ] || srun -n 16 lmp -in in.npt -log log.npt > screen.npt 2>&1
echo "### npt done $(date)"
srun -n 16 lmp -in in.sitefield -log log.sitefield > screen.sitefield 2>&1
LZ=$(awk '/zlo zhi/{printf "%.15f", $2-$1}' data.npt.lmp)
/mnt/home/lyuliyao/.conda/envs/aamd/bin/python check_site.py $LZ
/mnt/home/lyuliyao/.conda/envs/aamd/bin/python ../tools/site_profile.py site.dump --nbins 20 --out prof_site
echo "### done $(date)"
