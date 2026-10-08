#!/bin/bash --login
#SBATCH --job-name=sip_rstchk
#SBATCH --account=multiscaleml
#SBATCH --constraint=amd20
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/gs21/scratch/lyuliyao/salt_in_polymer/runs/restart_test/%x-%j.out
set -e
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
source /mnt/gs21/scratch/lyuliyao/salt_in_polymer/env_msu.sh
S=$SIP_ROOT/runs/pilot_T1.0_c0.04_one_both
W=$SIP_ROOT/runs/restart_test/fidelity
mkdir -p $W && cd $W
R=$SIP_ROOT/runs/restart_test/field_p00/restart.10000
ARGS="-in $SIP_ROOT/tests/in.restartcheck -var ff $SIP_ROOT/templates/ff.lmp \
      -var tabfile $SIP_ROOT/tools/solvation.table -var potfile $S/potentials/p00.lmp \
      -var n1 1000 -var n2 1000 -var pppm_acc 1.0e-4"
srun -n 32 $LMP -log log.straight $ARGS -var restart $R -var mode straight -var out straight.dump > /dev/null
srun -n 32 $LMP -log log.first    $ARGS -var restart $R -var mode first    -var out unused.dump > /dev/null
srun -n 32 $LMP -log log.second   $ARGS -var restart restart.mid -var mode second -var out resumed.dump > /dev/null
echo "### restart fidelity run done"
