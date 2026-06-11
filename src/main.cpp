#include "config.hpp"
#include "fvm_solver.hpp"
#include "geometry.hpp"
#include "lbm_solver.hpp"

#include <exception>
#include <iostream>
#include <mpi.h>
#include <string>

// 프로그램 진입점은 의도적으로 짧게 유지했다.
// 1) MPI 초기화
// 2) config/geometry 구성
// 3) solver.method에 따라 LBM 또는 FVM solver 실행
// 4) 예외 발생 시 모든 rank를 MPI_Abort로 종료
int main(int argc, char** argv) {
    int provided = 0;
    // OpenMP는 각 rank 내부에서만 사용하므로 MPI_THREAD_FUNNELED 수준이면 충분하다.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        // rank 0만 help/warning을 출력하지만 모든 rank가 같은 옵션을 파싱한다.
        ConfigResult cfg = ConfigParser::load(argc, argv, rank);
        if (rank == 0) {
            for (const auto& w : cfg.warnings) std::cout << "Config warning: " << w << '\n';
        }
        // geometry.file이 있으면 외부 primitive 파일을, 없으면 built-in F1 preset을 사용한다.
        Geometry2D geom = cfg.options.geometry_file.empty()
            ? Geometry2D::builtin(cfg.options)
            : Geometry2D::from_file(cfg.options, cfg.options.geometry_file);
        // 최종 검증 output은 lbm이지만 fvm solver도 같은 interface로 선택 가능하다.
        if (cfg.options.solver_method == "fvm") {
            FVMSolver solver(cfg.options, geom, MPI_COMM_WORLD);
            solver.run();
        } else {
            LBMSolver solver(cfg.options, geom, MPI_COMM_WORLD);
            solver.run();
        }
    } catch (const std::exception& e) {
        // 한 rank에서만 문제가 생겨도 병렬 프로그램이 hang되지 않도록 전체 communicator를 중단한다.
        if (rank == 0) std::cerr << "Fatal: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}
