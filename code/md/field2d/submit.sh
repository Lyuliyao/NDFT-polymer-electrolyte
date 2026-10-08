#!/bin/bash
# Two-dimensional test: potentials V(x, y) p50-p53 (tools/make_potential2d.py) at c = 0.04, both eps_r,
# C7 electrostatics, the production run lengths of each eps_r, template in.05_field2d; scavenger, one node.
cd /mnt/gs21/scratch/lyuliyao/salt_in_polymer
export SLURM_HINT=nomultithread
run(){  # $1 state dir, $2 ff dir, $3 restart, $4 ntrans, $5 nsteps
  env TEMPLATE=in.05_field2d FF=$2/ff.lmp TAB=$2/solvation.table SRCRESTART=$3 PPPM=1.0e-4 NTASKS=64 NNODES=1 MEMCPU=1G \
      NRST=250000 NDUMP=200 NREP=500 PARTITION=scavenger ACCOUNT=scavenger QOS=scavenger \
      CONSTRAINT="amd22|amd24" TLIM=3-00:00:00 ARRAY=50-53 ./run_state.sh field $1 $4 $5 | tail -1; }
run $PWD/field2d/eps75_c0.04 $PWD/bigbox/eps75_c0.04_x1 $PWD/bigbox/eps75_c0.04_x1/zero_field/restart.zf 1200000 11600000
run $PWD/field2d/eps2_c0.04 $PWD/bigbox/control_eps2_c0.04_C7 $PWD/eps2/runs/prod_T1.0_c0.04/zero_field/restart.zf 3200000 15200000
