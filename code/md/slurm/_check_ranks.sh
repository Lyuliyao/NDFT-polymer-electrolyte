# Guard against a broken MPI launcher.  With OpenMPI 5 + srun on this cluster,
# `srun -n N` starts N INDEPENDENT serial LAMMPS processes: each reports
# "1 MPI tasks", they all write the same files, and the job burns N times the
# SU while producing racing output.  Fail loudly instead.
check_ranks() {
  local log=$1 want=$2
  local got
  got=$(grep -m1 -oE "[0-9]+ MPI tasks" "$log" 2>/dev/null | awk '{print $1}')
  if [ "$got" != "$want" ]; then
    echo "FATAL: LAMMPS ran on '$got' MPI ranks but $want were requested." >&2
    echo "       The launcher is not forming a single communicator ($log)." >&2
    exit 1
  fi
  echo "[ranks ok] $log: $got MPI ranks"
}
