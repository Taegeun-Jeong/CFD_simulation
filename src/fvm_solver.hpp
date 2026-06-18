#pragma once

#include "config.hpp"
#include "geometry.hpp"

#include <fstream>
#include <mpi.h>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// FVMSolver
// ---------------------------------------------------------------------------
// Optional finite-volume comparison path.
// It follows projection-method Navier-Stokes with explicit LES update and
// retains an interface close to LBMSolver for run-time option parity.
class FVMSolver {
public:
    FVMSolver(const Options& opt, const Geometry2D& geom, MPI_Comm comm);
    void run();

private:
    // Immutable configuration and geometry reference.
    const Options opt_;
    const Geometry2D& geom_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int size_ = 1;

    // Decomposition metadata.
    int nx_ = 0;
    int ny_ = 0;
    int local_ny_ = 0;
    int y_start_ = 0;
    int lower_rank_ = MPI_PROC_NULL;
    int upper_rank_ = MPI_PROC_NULL;

    // Grid spacing, fixed physical timestep, and flow scales.
    double dx_ = 0.0;
    double dy_ = 0.0;
    double dt_ = 0.0;
    double nu_ = 0.0;
    double re_ = 0.0;
    double eta_ = 0.0;
    double ref_height_ = 1.0;

    // Cell-centered primary variables and auxiliaries.
    // u/v        : velocity components [m/s]
    // p          : pressure [Pa]
    // u_pred/u_star, v_pred/v_star : RK2 prediction/correction stages
    // rhs/nu_eff etc. : right-hand side and viscosity fields
    // phi_        : pressure correction potential
    std::vector<double> u_;
    std::vector<double> v_;
    std::vector<double> p_;
    std::vector<double> u_pred_;
    std::vector<double> v_pred_;
    std::vector<double> u_star_;
    std::vector<double> v_star_;
    std::vector<double> rhs_u_;
    std::vector<double> rhs_v_;
    std::vector<double> rhs_u2_;
    std::vector<double> rhs_v2_;
    std::vector<double> nu_eff_;
    std::vector<double> phi_;
    std::vector<double> phi_new_;
    std::vector<double> div_rhs_;
    std::vector<unsigned char> solid_;

    // Rank-0 drag history output.
    std::ofstream drag_csv_;

    // Flat indices with halo rows.
    [[nodiscard]] int cell(int ly, int x) const { return ly * nx_ + x; }
    [[nodiscard]] int global_y(int ly) const { return y_start_ + (ly - 1); }
    [[nodiscard]] double phys_x(int x) const { return static_cast<double>(x) * dx_; }
    [[nodiscard]] double phys_y_from_global(int gy) const { return static_cast<double>(gy) * dy_; }
    [[nodiscard]] bool valid_global_y(int gy) const { return gy >= 0 && gy < ny_; }
    [[nodiscard]] bool is_solid(int ly, int x) const { return solid_[cell(ly, x)] != 0; }

    void decompose();
    void initialize_fields();
    void initialize_outputs();
    void print_summary() const;

    // One-step FVM pipeline primitives.
    void exchange_scalar_rows(std::vector<double>& a, int tag_base) const;
    void apply_velocity_bc(std::vector<double>& u, std::vector<double>& v);
    void apply_pressure_bc(std::vector<double>& phi);
    void compute_les_viscosity(const std::vector<double>& u, const std::vector<double>& v);
    void compute_rhs(const std::vector<double>& u, const std::vector<double>& v,
                     std::vector<double>& ru, std::vector<double>& rv) const;
    void project_velocity();
    double pressure_poisson();

    // Output data preparation helpers (rank-local + global gather).
    std::vector<double> gather_scalar(const std::vector<double>& local) const;
    std::vector<double> local_scalar(const std::vector<double>& a) const;
    std::vector<double> local_solid() const;
    std::vector<double> local_vorticity() const;
    std::vector<double> local_nu_eff() const;

    void write_vtk(int step);
    void write_drag(int step);
};
