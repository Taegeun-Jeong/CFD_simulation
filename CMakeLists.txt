cmake_minimum_required(VERSION 3.18)
project(f1_les_cfd LANGUAGES CXX)

# C++17만 사용해서 cluster 환경에서도 별도 의존성 없이 빌드되게 한다.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 병렬화 핵심: rank 간 분할은 MPI, rank 내부 cell loop는 OpenMP가 담당한다.
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# 최종 실행 파일. solver/geometry/config/output 모듈을 분리해 구조를 명확히 했다.
add_executable(f1_cfd
    src/main.cpp
    src/config.cpp
    src/fvm_solver.cpp
    src/geometry.cpp
    src/lbm_solver.cpp
    src/vtk_writer.cpp
)

target_include_directories(f1_cfd PRIVATE src)
target_link_libraries(f1_cfd PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# 제출용 production build에서 성능을 확보하기 위해 O3와 기본 warning을 켠다.
target_compile_options(f1_cfd PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -O3>
)

# install target은 선택 사항이지만, examples까지 같이 배포할 수 있게 유지한다.
install(TARGETS f1_cfd RUNTIME DESTINATION bin)
install(DIRECTORY examples/ DESTINATION share/f1_les_cfd/examples)
