cmake_minimum_required(VERSION 3.18)
project(f1_les_cfd LANGUAGES CXX)

# Build with C++17 only to stay dependency-free in cluster environments.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Parallel core: MPI splits work across ranks, OpenMP handles cell loops inside each rank.
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Final executable: separated solver/geometry/config/output modules for clear structure.
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

# Enable O3 and baseline warnings for production performance.
target_compile_options(f1_cfd PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -O3>
)

# Install target is optional but kept to allow shipping examples as well.
install(TARGETS f1_cfd RUNTIME DESTINATION bin)
install(DIRECTORY examples/ DESTINATION share/f1_les_cfd/examples)
