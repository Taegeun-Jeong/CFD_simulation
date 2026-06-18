#!/usr/bin/env bash
set -euo pipefail

# Utility script to run three built-in F1 geometries with the same solver/mesh/time settings.
# Sedan/SUV use custom geometry files, so run them with the per-case mpirun commands in README.
exe=${1:-./build/f1_cfd}
shift || true
ranks=${RANKS:-4}
steps=${STEPS:-120000}
threads=${OMP_NUM_THREADS:-4}
launcher=${MPI_LAUNCHER:-mpirun}
export OMP_NUM_THREADS="$threads"
extra_args=("$@")

# fork+shared-memory launch is stable for local Intel MPI outside Slurm allocation.
# Use MPI_LAUNCHER=srun in cluster mode if you want to run with srun.
if [[ "$(basename "${launcher%% *}")" == mpirun || "$(basename "${launcher%% *}")" == mpiexec ]]; then
  export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-fork}
  export I_MPI_FABRICS=${I_MPI_FABRICS:-shm}
fi

# The three F1 configs are saved in separate directories/prefixes under output/f1.
configs=(examples/f1_demo.ini examples/f1_low_drag.ini examples/f1_high_downforce.ini)
for cfg in "${configs[@]}"; do
  echo "=== Running $cfg with $ranks MPI ranks, $threads OpenMP threads, steps=$steps ==="
  # shellcheck disable=SC2086 # Allow launchers with spaces such as MPI_LAUNCHER="srun --mpi=pmi2"
  $launcher -n "$ranks" "$exe" --config "$cfg" --set time.steps="$steps" "${extra_args[@]}"
done

# Read the last line of each CSV to compare final Cd/Cl quickly.
printf '\nFinal drag coefficients:\n'
for csv in output/f1/high_re_generic/f1_high_re_generic_drag.csv output/f1/high_re_low_drag/f1_high_re_low_drag_drag.csv output/f1/high_re_high_downforce/f1_high_re_high_downforce_drag.csv; do
  if [[ -f "$csv" ]]; then
    tail -n 1 "$csv" | awk -F, -v file="$csv" '{printf "%s: step=%s Cd=%s Cl=%s\n", file, $1, $5, $6}'
  fi
done
