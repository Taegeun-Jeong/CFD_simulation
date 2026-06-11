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
// 확장/비교용 finite-volume projection LES solver.
// 현재 최종 제출 결과는 안정 검증된 LBM 경로에서 생성했지만, FVM 구조를
// 별도로 유지해 solver.method=fvm으로 실험할 수 있게 했다.
class FVMSolver {
public:
    FVMSolver(const Options& opt, const Geometry2D& geom, MPI_Comm comm);
    void run();

private:
    // 입력 옵션/형상과 MPI 실행 정보.
    const Options opt_;
    const Geometry2D& geom_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int size_ = 1;

    int nx_ = 0;
    int ny_ = 0;
    int local_ny_ = 0;
    int y_start_ = 0;
    int lower_rank_ = MPI_PROC_NULL;
    int upper_rank_ = MPI_PROC_NULL;

    // 물리 격자, timestep, 점성, Reynolds/Kolmogorov 추정값.
    double dx_ = 0.0;
    double dy_ = 0.0;
    double dt_ = 0.0;
    double nu_ = 0.0;
    double re_ = 0.0;
    double eta_ = 0.0;
    double ref_height_ = 1.0;

    // cell-centered FVM field. predictor/projection 단계 때문에 보조 배열을 분리했다.
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

    // rank 0의 force history 출력 파일.
    std::ofstream drag_csv_;

    // halo row를 포함한 local 배열 index helper.
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

    // FVM 한 step을 구성하는 주요 연산: boundary, LES 점성, RHS, pressure projection.
    void exchange_scalar_rows(std::vector<double>& a, int tag_base) const;
    void apply_velocity_bc(std::vector<double>& u, std::vector<double>& v);
    void apply_pressure_bc(std::vector<double>& phi);
    void compute_les_viscosity(const std::vector<double>& u, const std::vector<double>& v);
    void compute_rhs(const std::vector<double>& u, const std::vector<double>& v,
                     std::vector<double>& ru, std::vector<double>& rv) const;
    void project_velocity();
    double pressure_poisson();

    // 출력 및 force 계산을 위한 rank 0 gather helper.
    std::vector<double> gather_scalar(const std::vector<double>& local) const;
    std::vector<double> local_scalar(const std::vector<double>& a) const;
    std::vector<double> local_solid() const;
    std::vector<double> local_vorticity() const;
    std::vector<double> local_nu_eff() const;

    void write_vtk(int step);
    void write_drag(int step);
};
