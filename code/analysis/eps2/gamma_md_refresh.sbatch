#!/bin/bash --login
# Recompute MD Gamma at every eps_r = 2 concentration after the c = 0.01 zero
# field was extended to 1e5 tau; the 2e4-tau version is kept as gamma_md_20k.json.
#SBATCH --job-name=sip_gammamd
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2/runs/learn/%x-%j.out
set -e
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_eps2
export SIP_ROOT=$R JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
I=$R/runs/learn/interpretation
[ -f $I/gamma_md_20k.json ] || cp -p $I/gamma_md.json $I/gamma_md_20k.json
cd $R/code
/mnt/home/lyuliyao/.conda/envs/heat/bin/python -m learn.gamma_md --concs 0.01 0.02 0.04 0.05 0.06 0.08 2>&1 | grep -v "plugin\|cuInit\|cuda"
echo "### done $(date)"
