#!/bin/bash --login
#SBATCH --job-name=lj_summary
#SBATCH --account=multiscaleml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj/runs/learn/%x-%j.out
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj
export SIP_ROOT=$R JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd $R/code && /mnt/home/lyuliyao/.conda/envs/heat/bin/python $R/figs/lj_arch_summary.py 2>&1 | grep -v "plugin\|cuInit\|cuda\|RuntimeWarning\|ax.plot\|r = S"
