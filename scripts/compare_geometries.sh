#!/usr/bin/env bash
set -euo pipefail

# F1 built-in geometry 3종을 같은 solver/mesh/time setting으로 연속 실행하는 편의 스크립트.
# sedan/SUV는 custom geometry 파일을 쓰므로 README의 개별 mpirun 명령으로 실행한다.
exe=${1:-./build/f1_cfd}
shift || true
ranks=${RANKS:-4}
steps=${STEPS:-120000}
threads=${OMP_NUM_THREADS:-4}
launcher=${MPI_LAUNCHER:-mpirun}
export OMP_NUM_THREADS="$threads"
extra_args=("$@")

# Slurm allocation 밖의 local Intel MPI에서는 fork+shared-memory launch가 안정적이다.
# cluster 환경에서 srun을 쓰려면 MPI_LAUNCHER=srun처럼 바꿔 실행하면 된다.
if [[ "$(basename "${launcher%% *}")" == mpirun || "$(basename "${launcher%% *}")" == mpiexec ]]; then
  export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-fork}
  export I_MPI_FABRICS=${I_MPI_FABRICS:-shm}
fi

# 세 F1 config는 output/f1 아래 서로 다른 디렉터리/prefix로 결과를 저장한다.
configs=(examples/f1_demo.ini examples/f1_low_drag.ini examples/f1_high_downforce.ini)
for cfg in "${configs[@]}"; do
  echo "=== Running $cfg with $ranks MPI ranks, $threads OpenMP threads, steps=$steps ==="
  # shellcheck disable=SC2086 # MPI_LAUNCHER="srun --mpi=pmi2"처럼 공백이 있는 launcher를 허용
  $launcher -n "$ranks" "$exe" --config "$cfg" --set time.steps="$steps" "${extra_args[@]}"
done

# 각 CSV 마지막 줄을 읽어 최종 Cd/Cl을 빠르게 비교한다.
printf '\nFinal drag coefficients:\n'
for csv in output/f1/high_re_generic/f1_high_re_generic_drag.csv output/f1/high_re_low_drag/f1_high_re_low_drag_drag.csv output/f1/high_re_high_downforce/f1_high_re_high_downforce_drag.csv; do
  if [[ -f "$csv" ]]; then
    tail -n 1 "$csv" | awk -F, -v file="$csv" '{printf "%s: step=%s Cd=%s Cl=%s\n", file, $1, $5, $6}'
  fi
done
