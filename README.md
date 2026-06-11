# F1 / Avante / Tucson 2D LES CFD Simulation (MPI + OpenMP)

이 프로젝트는 **자동차를 옆에서 바라본 2차원(side-view) 형상** 주변의 비정상 유동을 계산하는 C++ CFD 코드입니다. 현재 최종 output은 세 형상 **F1, Hyundai Avante, Hyundai Tucson**을 같은 mesh/time setting과 같은 **Reynolds number `Re=800`**에서 비교한 결과입니다.

> `output/old/`에는 이전에 보존한 결과만 남겨두었습니다: `output/old/f1/high_re_generic`, `output/old/sedan`. 새 최종 비교 결과는 `output/f1`, `output/avante`, `output/tucson`입니다.

## 평가 포인트 요약

### Performance: efficiency & accuracy

- **Hybrid 병렬화**: MPI y-domain decomposition + OpenMP cell-loop parallelism. 최종 실행은 `4 MPI ranks x 4 OpenMP threads = 16 cores`로 수행했습니다.
- **메모리 효율**: field를 1차원 contiguous vector에 저장하고, MPI 통신은 인접 rank와 halo row만 교환합니다.
- **출력 비용 제어**: VTK/CSV 출력은 매 step이 아니라 `output_interval=240`, 즉 `0.02 s`마다 수행합니다.
- **수치 정확도**: 최종 검증 경로는 D2Q9 LBM 기반 LES입니다. 저 Mach/incompressible 한계에서 2차 정확도 특성을 갖고, vorticity/strain-rate는 중앙차분 기반으로 계산합니다.
- **LES 해상도 고려**: Kolmogorov estimate `eta ≈ L Re^(-3/4)`를 출력하며, 최종 mesh는 `1081 x 271`, `dx=dy=0.0166667 m`입니다.
- **고정 timestep**: CFL로 시작 시 `dt`를 한 번만 계산하고 run 중에는 바꾸지 않아 비교 재현성을 유지합니다.

### Algorithm: neatness & creativity

- **형상 교체형 설계**: F1은 built-in preset, Avante/Tucson은 side-view 사진을 참고해 만든 `polygon/circle` 기반 외부 geometry 파일을 사용합니다.
- **공정 비교 조건**: 세 형상을 모두 같은 `Re=800`, 같은 domain, 같은 mesh, 같은 inlet velocity, 같은 output interval로 실행했습니다.
- **LES 모델**: Smagorinsky LES 모델을 사용하며 `Cs`, filter width, `tau_min/tau_max`는 config에서 변경 가능합니다.
- **항력/양력 계산**: LBM bounce-back momentum exchange를 이용해 force를 계산하고, `Cd`, `Cl`, `N/m` 단위 force를 CSV로 저장합니다.
- **확장성**: `solver.method=fvm`으로 finite-volume/projection LES solver도 선택 가능하지만, 최종 결과는 안정 검증된 LBM 경로로 생성했습니다.

### Readability: structure & comments

- `src/config.*`: INI/CLI 옵션 파싱, 기본값 생성, mesh/Re/CFL sanity check.
- `src/geometry.*`: F1 preset 생성, custom geometry 파일 파싱, solid mask 판정, geometry VTK 출력.
- `src/lbm_solver.*`: 최종 LBM-LES solver, MPI/OpenMP 병렬 계산, VTK/CSV 출력.
- `src/fvm_solver.*`: 실험용 FVM projection LES solver.
- `src/vtk_writer.*`: ParaView용 legacy structured-points VTK writer.
- 모든 C++ 소스, CMake, 실행 스크립트, 예제 config/geometry에는 한글 주석을 추가했습니다.

## 빌드 방법

프로젝트 루트에서 실행합니다.

```bash
cd /home/taegeun/project/Parallel_Computing/Final_Project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## MPI + OpenMP 직접 실행 방법

이 코드는 hybrid parallel 방식입니다. 직접 실행할 때는 아래 두 값을 정하면 됩니다.

- **MPI process/rank 수**: `mpirun -n <rank수>` 또는 Slurm에서는 `srun -n <rank수>`
- **OpenMP thread 수**: `OMP_NUM_THREADS=<thread수>`

이번 최종 결과에서는 사용 가능 core를 넉넉히 잡아 **4 MPI ranks x 4 OpenMP threads = 총 16 cores**로 실행했습니다.

일반 형식은 다음과 같습니다.

```bash
cd /home/taegeun/project/Parallel_Computing/Final_Project
I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm OMP_NUM_THREADS=4 \
mpirun -n 4 ./build/f1_cfd --config <config-file.ini>
```

여기서 `I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm`는 이 로컬 Intel MPI 환경에서 `mpirun`을 안정적으로 띄우기 위해 사용했습니다. Slurm allocation 내부에서는 보통 다음처럼 `srun`으로 실행할 수 있습니다.

```bash
cd /home/taegeun/project/Parallel_Computing/Final_Project
OMP_NUM_THREADS=4 srun -n 4 ./build/f1_cfd --config examples/f1.ini
```

## 최종 3개 예제 실행 명령

세 case 모두 같은 `Re=800`, 같은 `steps=120000`, 같은 `output_interval=240`입니다. **이번 최종 output을 만들 때 실제로 사용한 3개 명령어**는 다음과 같습니다.

```bash
cd /home/taegeun/project/Parallel_Computing/Final_Project

I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm OMP_NUM_THREADS=4 \
mpirun -n 4 ./build/f1_cfd --config examples/f1.ini

I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm OMP_NUM_THREADS=4 \
mpirun -n 4 ./build/f1_cfd --config examples/avante.ini

I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm OMP_NUM_THREADS=4 \
mpirun -n 4 ./build/f1_cfd --config examples/tucson.ini
```

## ParaView에서 열 파일

최종 설정은 `dt=8.333333e-5 s`, `output_interval=240`이므로 **0.02초마다 VTK 1개**가 저장됩니다. `steps=120000`이므로 총 물리 시간은 **10초**이고, 각 case마다 time VTK `501`개(`t=0.00`부터 `t=10.00 s`)와 geometry VTK `1`개가 생성됩니다.

대표 최종 파일:

```text
output/f1/f1_0120000.vtk
output/avante/avante_0120000.vtk
output/tucson/tucson_0120000.vtk
```

항력/양력 CSV:

```text
output/f1/f1_drag.csv
output/avante/avante_drag.csv
output/tucson/tucson_drag.csv
```

ParaView field:

- `velocity_m_s`: 물리 단위 속도 벡터 `[m/s]`
- `vorticity_s`: 와도 `[1/s]`
- `solid`: 차량/벽면 solid mask
- `nu_eff_lu`: LBM lattice 단위 LES 유효 점성계수

## 최종 3개 예제의 물리적/수치적 파라미터

공통 parameter:

| 항목 | 값 |
| --- | --- |
| Domain size | `18.0 m x 4.5 m` |
| Mesh | `1081 x 271` |
| Grid spacing | `dx = dy = 0.0166667 m` |
| Inlet velocity | `30.0 m/s` |
| Density | `1.225 kg/m^3` |
| Dynamic viscosity option | `mu = 0.0`, 즉 `nu = U L / Re` |
| Reynolds number | `Re = 800` for all 3 cases |
| CFL | `0.15` |
| Fixed timestep | `dt = 8.333333e-5 s` |
| Total time | `120000 steps = 10.0 s` |
| Output / drag interval | `240 steps = 0.02 s` |
| Log interval | `1200 steps = 0.1 s` |
| LES model | Smagorinsky, `Cs = 0.30`, `filter_width_cells = 1.0` |
| LBM relaxation limits | `tau_min = 0.5001`, `tau_max = 2.5` |
| Parallel run | `4 MPI ranks x 4 OpenMP threads` |

Case별 parameter:

| Case | Config | Geometry | Re | `nu` `[m^2/s]` | `tau0` | `eta` `[m]` | `dx/eta` | `car_length` / `car_height` `[m]` | Output |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F1 | `examples/f1.ini` | `generic_f1_2026` preset | 800 | 0.20625 | 0.685625 | 0.03656 | 0.456 | 5.50 / 1.15 | `output/f1` |
| Avante | `examples/avante.ini` | `examples/avante_geometry.geom` | 800 | 0.176625 | 0.658963 | 0.03131 | 0.532 | 4.71 / 1.42 | `output/avante` |
| Tucson | `examples/tucson.ini` | `examples/tucson_geometry.geom` | 800 | 0.17400 | 0.656600 | 0.03085 | 0.540 | 4.64 / 1.665 | `output/tucson` |

## 현재 최종 결과 요약

최종 step(`t=10 s`)의 drag CSV 마지막 행 기준입니다.

| Case | Final Cd | Final Cl | Drag `[N/m]` | Lift `[N/m]` |
| --- | ---: | ---: | ---: | ---: |
| F1 | 6.44376 | 118.228 | 4386.87 | 80488.7 |
| Avante | 3.59096 | 11.7589 | 2702.04 | 8848.02 |
| Tucson | 3.96510 | 10.0166 | 3628.37 | 9165.92 |

이 값들은 2D side-view, simplified/photo-traced geometry, LES/LBM 설정에서 얻은 비교용 결과입니다. 실제 3D 자동차 풍동 Cd/Cl과 직접 비교하는 목적이 아니라, 같은 solver/mesh/BC/Re 조건에서 형상 변화가 force history와 wake 구조에 주는 차이를 비교하기 위한 값입니다.

## Geometry 파일 형식

`geometry.file`을 지정하면 built-in preset 대신 외부 primitive 파일을 사용합니다.

```text
name optional_name
polygon x1 y1 x2 y2 x3 y3 ...
circle cx cy r
rect xmin ymin xmax ymax
```

Avante/Tucson geometry는 공식 side-view 사진을 참고한 외곽선 단순화이며, 실제 CAD가 아닙니다.
