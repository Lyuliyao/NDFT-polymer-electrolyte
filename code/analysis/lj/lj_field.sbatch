#!/bin/bash --login
# One LJ field run per array task (tag p<task>), from the state point's equilibrated
# restart, then its profiles and acceptance.
#   STATEDIR=<dir> sbatch --export=ALL --array=0-21 lj_field.sbatch
#SBATCH --job-name=lj_fld
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20|amd22|amd24
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj/runs/%x-%A_%a.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
: ${STATEDIR:?}; : ${NTRANS:=200000}; : ${NSTEPS:=2000000}; : ${NBLOCK:=125000}
NP=${SLURM_NTASKS:-16}
TAG=$(printf "p%02d" ${SLURM_ARRAY_TASK_ID:-0})
D=$STATEDIR/field_$TAG
mkdir -p $D && cd $D
test -f $STATEDIR/potentials/$TAG.lmp || { echo "missing potential $TAG"; exit 1; }
echo "### $(date): $TAG in $STATEDIR on $(hostname)"
if ! grep -q "^Total wall time" log.field 2>/dev/null; then
  rm -f prof_cat.dat prof_ani.dat current.dat
  srun -n $NP $LMP -log log.field -in $R/templates/in.lj_field -var ff $R/templates/ff_lj.lmp \
      -var restart $STATEDIR/equil/restart.eq -var potfile $STATEDIR/potentials/$TAG.lmp \
      -var ntrans $NTRANS -var nsteps $NSTEPS -var nblock $NBLOCK > /dev/null
fi
grep "Performance" log.field | tail -1
$PY $R/tools/lj_profiles.py --run $D --pot $STATEDIR/potentials/$TAG.json
echo "### done $(date)"
