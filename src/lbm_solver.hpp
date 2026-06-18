#pragma once

#include "config.hpp"
#include "geometry.hpp"

#include <array>
#include <fstream>
#include <mpi.h>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LBMSolver
// ---------------------------------------------------------------------------
// Production path: D2Q9 LBM + Smagorinsky LES.
// - MPI domain decomposition in y-direction splits rows across ranks.
// - OpenMP parallelizes rank-local cell operations.
// - Arrays are 1D flat buffers with halo rows at [0] and [local_ny+1].
class LBMSolver {
public:
    LBMSolver(const Options& opt, const Geometry2D& geom, MPI_Comm comm);
    void run();

private:
    // D2Q9 model constants.
    // cx/cy: lattice velocity vectors in order [0..8], opp: opposite direction indices,
    // wt: lattice weights for equilibrium polynomial.
    static constexpr int Q = 9;
    static constexpr std::array<int, Q> cx{{0, 1, 0, -1, 0, 1, -1, -1, 1}};
    static constexpr std::array<int, Q> cy{{0, 0, 1, 0, -1, 1, 1, -1, -1}};
    static constexpr std::array<int, Q> opp{{0, 3, 4, 1, 2, 7, 8, 5, 6}};
    static constexpr std::array<double, Q> wt{{4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                                               1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0}};

    // Immutable solver configuration and geometry reference.
    const Options opt_;       // options are copied once and never modified in solver
    const Geometry2D& geom_;  // geometry pre-built before LBMSolver construction
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int size_ = 1;

    // Global and local domain decomposition metadata.
    int nx_ = 0;
    int ny_ = 0;
    int local_ny_ = 0;
    int y_start_ = 0;
    int lower_rank_ = MPI_PROC_NULL;
    int upper_rank_ = MPI_PROC_NULL;

    // Physical and lattice-scale parameters.
    // dx_,dy_,dt_ are physical dimensions/time.
    // u_lu_, nu_lu_, tau* are lattice-space relaxation/scaling quantities.
    double dx_ = 0.0;
    double dy_ = 0.0;
    double dt_ = 0.0;
    double u_lu_ = 0.0;
    double nu_phys_ = 0.0;
    double nu_lu_ = 0.0;
    double tau0_ = 0.0;
    double re_ = 0.0;
    double kolmogorov_eta_ = 0.0;
    double ref_height_ = 1.0;

    // Core fields:
    // f_ / post_ / f_next_: distribution functions (Q directions, flat storage)
    // rho_/ux_/uy_: macro fields in lattice units.
    // tau_eff_: LES-modified relaxation time.
    // solid_: mask for no-slip immersed geometry, unsigned char is compact and cache-friendly.
    std::vector<double> f_;
    std::vector<double> post_;
    std::vector<double> f_next_;
    std::vector<double> rho_;
    std::vector<double> ux_;
    std::vector<double> uy_;
    std::vector<double> tau_eff_;
    std::vector<unsigned char> solid_;

    // Only rank 0 owns this output stream; other ranks still compute drag and reduce.
    std::ofstream drag_csv_;

    // Flat index helpers.
    // cell(ly, x) maps interior+halo (ly in [0, local_ny+1]) to [0, local_nx*local_ny+2*nx).
    // idx(ly,x,q) maps distribution direction q on cell.
    [[nodiscard]] int cell(int ly, int x) const { return ly * nx_ + x; }
    [[nodiscard]] int idx(int ly, int x, int q) const { return (cell(ly, x) * Q) + q; }
    [[nodiscard]] int global_y(int ly) const { return y_start_ + (ly - 1); }
    [[nodiscard]] double phys_x(int x) const { return static_cast<double>(x) * dx_; }
    [[nodiscard]] double phys_y_from_global(int gy) const { return static_cast<double>(gy) * dy_; }
    [[nodiscard]] bool in_global_y(int gy) const { return gy >= 0 && gy < ny_; }
    [[nodiscard]] bool is_solid(int ly, int x) const { return solid_[cell(ly, x)] != 0; }

    // Life-cycle helpers.
    void decompose();
    void initialize_fields();
    void initialize_outputs();
    void print_summary(const std::vector<std::string>& runtime_warnings) const;

    // Algorithm blocks for each time-step stage.
    // equilibrium -> macro from f_ -> LES tau -> collision -> stream/bounce -> BCs.
    double equilibrium(int q, double rho, double ux, double uy) const;
    void compute_macros();
    void exchange_scalar_rows(std::vector<double>& a, int tag_base);
    void exchange_dist_rows(std::vector<double>& a, int tag_base);
    void compute_les_tau();
    void collide();
    std::array<double, 2> stream_and_bounce();
    void apply_boundaries();

    // Gather helpers: convert rank-local fields to rank-0 structured global arrays for VTK.
    std::vector<double> gather_scalar(const std::vector<double>& local) const;
    std::vector<double> make_local_scalar_rho() const;
    std::vector<double> make_local_scalar_ux_phys() const;
    std::vector<double> make_local_scalar_uy_phys() const;
    std::vector<double> make_local_scalar_solid() const;
    std::vector<double> make_local_scalar_nu_eff() const;
    std::vector<double> make_local_scalar_vorticity() const;

    void write_vtk(int step);
    void write_drag(int step, double local_fx, double local_fy);
    void log_step(int step, double global_fx, double global_fy) const;
    void ensure_finite(int step) const;
};
