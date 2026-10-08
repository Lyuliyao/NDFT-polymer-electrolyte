# Environment for the salt-in-polymer CG-MD campaign on MSU HPCC (amd20 nodes).
# Counterpart of env.sh (PSC Bridges-2).  Source before running LAMMPS or analysis.

module purge >/dev/null 2>&1
module load GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0 FFTW/3.3.10-GCC-12.3.0 >/dev/null 2>&1

export SIP_SITE=msu
export SIP_ROOT=/mnt/gs21/scratch/lyuliyao/salt_in_polymer
# Hall-group bornsolv fork (LAMMPS 29Oct2020), built by build/lammps_bornsolv/build_amd20.sh
export LMP_BS=$SIP_ROOT/build/lammps_bornsolv/build-amd20/lmp_bornsolv
export LMP=$LMP_BS

# Analysis python: the tools only need numpy.
export PYTHONNOUSERSITE=1
export SIP_PY=/mnt/home/lyuliyao/.conda/envs/mdmd/bin/python
