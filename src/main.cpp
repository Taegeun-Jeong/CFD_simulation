#include "config.hpp"
#include "fvm_solver.hpp"
#include "geometry.hpp"
#include "lbm_solver.hpp"

#include <exception>
#include <iostream>
#include <mpi.h>
#include <string>

// Minimal driver entry point:
// 1) Initialize MPI with thread level suitable for OpenMP inside each rank,
// 2) parse options and construct geometry,
// 3) run selected solver,
// 4) convert all thrown exceptions into a controlled MPI abort.
int main(int argc, char** argv) {
    int provided = 0;
    // FUNNEL model is enough because OpenMP work is nested in the rank-local loop body,
    // and MPI calls are only made from the main thread.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        // Parse configuration on all ranks so overrides and derived values are identical.
        ConfigResult cfg = ConfigParser::load(argc, argv, rank);

        // Only rank 0 prints non-fatal warnings to avoid duplicated logs.
        if (rank == 0) {
            for (const auto& w : cfg.warnings) std::cout << "Config warning: " << w << '\n';
        }

        // Use preset geometry when geometry.file is empty, otherwise read external file.
        // This makes geometry replacement easy during batch studies.
        Geometry2D geom = cfg.options.geometry_file.empty()
            ? Geometry2D::builtin(cfg.options)
            : Geometry2D::from_file(cfg.options, cfg.options.geometry_file);

        // Route solver creation by method string.
        // Both solvers expose the same run() API for consistent launch behavior.
        if (cfg.options.solver_method == "fvm") {
            FVMSolver solver(cfg.options, geom, MPI_COMM_WORLD);
            solver.run();
        } else {
            LBMSolver solver(cfg.options, geom, MPI_COMM_WORLD);
            solver.run();
        }
    } catch (const std::exception& e) {
        // If any rank throws, abort all ranks to prevent indefinite wait in MPI collectives.
        if (rank == 0) std::cerr << "Fatal: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}
