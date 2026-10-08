# Environment for the salt-in-polymer CG-MD campaign (PSC Bridges-2)
# source this before running LAMMPS or the analysis tools.

# On MSU HPCC the MSU environment applies instead (env_msu.sh).
if [ -d /mnt/gs21/scratch ]; then
  source "$(dirname "${BASH_SOURCE[0]}")/env_msu.sh"
  return 0 2>/dev/null || exit 0
fi
export SIP_SITE=bridges

module purge >/dev/null 2>&1
module load python/3.8.6 openmpi/5.0.8-gcc13.3.1 intel-mkl/2023.2.0 >/dev/null 2>&1

export LMP=/opt/packages/LAMMPS/lammps-10Sep2025/build-RM-gcc13.3.1/lmp
export SIP_ROOT=/ocean/projects/mth210003p/lyuliyao/salt_in_polymer

# Analysis python -- separate from the LAMMPS runtime python.
# PYTHONNOUSERSITE is essential: ~/.local holds numpy 2.2.6, which shadows the
# conda numpy 1.26.4 that this anaconda's matplotlib and scipy were built
# against, and breaks both.  Disabling user-site restores a consistent stack.
export PYTHONNOUSERSITE=1
export SIP_PY=/opt/packages/anaconda3-2024.10-1/bin/python3
