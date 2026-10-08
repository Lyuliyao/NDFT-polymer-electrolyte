#!/bin/bash --login
# One LJ state point: equilibration and the zero-field run, then its analysis
# (STAGE=equil stops after the equilibration, STAGE=zero expects it done).
#   STATEDIR=<dir> RHO=0.40 NHALF=2765 SEED=40 sbatch --export=ALL lj_state.sbatch
#SBATCH --job-name=lj_state
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20|amd22|amd24
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lyuliyao@msu.edu
#SBATCH --output=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj/runs/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
R=/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_lj
PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
: ${STATEDIR:?}; : ${RHO:?}; : ${NHALF:?}; : ${SEED:?}
: ${NEQUIL:=200000}; : ${ZF_NTRANS:=20000}; : ${ZF_NSTEPS:=20000000}; : ${ZF_NDUMP:=2000}
: ${STAGE:=both}                          # equil | zero | both
NP=${SLURM_NTASKS:-16}
mkdir -p $STATEDIR/equil $STATEDIR/zero_field
echo "### $(date): rho $RHO, $NHALF per label, seed $SEED on $(hostname)"
cd $STATEDIR/equil
test -f restart.eq || srun -n $NP $LMP -log log.equil -in $R/templates/in.lj_equil -var ff $R/templates/ff_lj.lmp \
    -var nhalf $NHALF -var seed $SEED -var nequil $NEQUIL -var out restart.eq > /dev/null
grep -A1 "^Loop time" log.equil | tail -1
[ "$STAGE" = equil ] && { echo "### equilibrated $(date)"; exit 0; }
cd $STATEDIR/zero_field
test -f restart.zf || srun -n $NP $LMP -log log.zero -in $R/templates/in.lj_zero -var ff $R/templates/ff_lj.lmp \
    -var restart $STATEDIR/equil/restart.eq -var ntrans $ZF_NTRANS -var nsteps $ZF_NSTEPS -var ndump $ZF_NDUMP \
    -var out restart.zf > /dev/null
grep "Performance" log.zero | tail -1
$PY $R/tools/lj_zerofield.py --zf $STATEDIR/zero_field --rho $RHO
echo "### done $(date)"
