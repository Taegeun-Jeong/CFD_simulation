#include "fvm_solver.hpp"

#include "vtk_writer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// 이 파일은 LBM과 별도로 둔 finite-volume/projection LES solver이다.
// 최종 output은 LBM으로 만들었지만, FVM/FDM/FEM 계열 확장을 고려해
// cell-centered FVM 구조, pressure projection, Smagorinsky LES를 구현해 두었다.

namespace {

// MPI_Gatherv 출력 gather용 count 계산.
std::vector<int> counts_for_y(int ny, int nx, int size) {
    std::vector<int> counts(size, 0);
    const int base = ny / size;
    const int rem = ny % size;
    for (int r = 0; r < size; ++r) counts[r] = (base + (r < rem ? 1 : 0)) * nx;
    return counts;
}

// gather displacement 계산.
std::vector<int> displs_from_counts(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size(), 0);
    for (std::size_t i = 1; i < counts.size(); ++i) displs[i] = displs[i - 1] + counts[i - 1];
    return displs;
}

// VTK step 파일명 생성 helper.
std::string step_name(const std::string& dir, const std::string& prefix, int step) {
    std::ostringstream path;
    path << dir << '/' << prefix << '_' << std::setw(7) << std::setfill('0') << step << ".vtk";
    return path.str();
}

} // 익명 namespace

// constructor에서 mesh 분해, 물리 파라미터, 배열 할당, 초기 조건을 모두 준비한다.
FVMSolver::FVMSolver(const Options& opt, const Geometry2D& geom, MPI_Comm comm)
    : opt_(opt), geom_(geom), comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    nx_ = opt_.nx;
    ny_ = opt_.ny;
    if (ny_ < size_) throw std::runtime_error("Number of MPI ranks must not exceed mesh.ny");
    decompose();

    // FVM도 timestep은 CFL 기반으로 시작 시 한 번만 계산하고 run 중 고정한다.
    dx_ = opt_.domain_length / static_cast<double>(nx_ - 1);
    dy_ = opt_.domain_height / static_cast<double>(ny_ - 1);
    dt_ = opt_.cfl * std::min(dx_, dy_) / opt_.inlet_velocity; // 전체 run 동안 고정
    // mu가 있으면 nu=mu/rho, 없으면 Re로부터 nu=U*L/Re 계산.
    if (opt_.dynamic_viscosity > 0.0) {
        nu_ = opt_.dynamic_viscosity / opt_.density;
        re_ = opt_.inlet_velocity * opt_.car_length / nu_;
    } else {
        re_ = opt_.reynolds;
        nu_ = opt_.inlet_velocity * opt_.car_length / re_;
    }
    eta_ = opt_.car_length * std::pow(std::max(re_, 1.0), -0.75);
    ref_height_ = geom_.height() > 0.0 ? geom_.height() : opt_.car_height;

    // halo 2줄을 포함한 cell-centered 배열 할당.
    const std::size_t n = static_cast<std::size_t>(local_ny_ + 2) * static_cast<std::size_t>(nx_);
    u_.assign(n, opt_.inlet_velocity);
    v_.assign(n, 0.0);
    p_.assign(n, 0.0);
    u_pred_.assign(n, 0.0);
    v_pred_.assign(n, 0.0);
    u_star_.assign(n, 0.0);
    v_star_.assign(n, 0.0);
    rhs_u_.assign(n, 0.0);
    rhs_v_.assign(n, 0.0);
    rhs_u2_.assign(n, 0.0);
    rhs_v2_.assign(n, 0.0);
    nu_eff_.assign(n, nu_);
    phi_.assign(n, 0.0);
    phi_new_.assign(n, 0.0);
    div_rhs_.assign(n, 0.0);
    solid_.assign(n, 0);

    initialize_fields();
    initialize_outputs();
}

// y 방향 1차원 MPI domain decomposition.
void FVMSolver::decompose() {
    const int base = ny_ / size_;
    const int rem = ny_ % size_;
    local_ny_ = base + (rank_ < rem ? 1 : 0);
    y_start_ = rank_ * base + std::min(rank_, rem);
    lower_rank_ = (y_start_ > 0) ? rank_ - 1 : MPI_PROC_NULL;
    upper_rank_ = (y_start_ + local_ny_ < ny_) ? rank_ + 1 : MPI_PROC_NULL;
}

// 초기 속도장과 solid mask 생성. perturbation은 wake 발달을 돕기 위한 작은 교란이다.
void FVMSolver::initialize_fields() {
    for (int ly = 0; ly < local_ny_ + 2; ++ly) {
        const int gy = global_y(ly);
        const bool valid_y = valid_global_y(gy);
        const double y = valid_y ? phys_y_from_global(gy) : -1.0;
        for (int x = 0; x < nx_; ++x) {
            const double px = phys_x(x);
            const bool solid = valid_y && geom_.contains(px, y);
            const int c = cell(ly, x);
            solid_[c] = solid ? 1 : 0;
            const double phase = 2.0 * 3.14159265358979323846 *
                (static_cast<double>(x) / std::max(1, nx_ - 1) +
                 static_cast<double>(std::max(gy, 0)) / std::max(1, ny_ - 1));
            u_[c] = solid ? 0.0 : opt_.inlet_velocity * (1.0 + opt_.perturbation * std::sin(phase));
            v_[c] = solid ? 0.0 : opt_.inlet_velocity * opt_.perturbation * 0.1 * std::cos(phase);
        }
    }
    apply_velocity_bc(u_, v_);
    exchange_scalar_rows(u_, 1000);
    exchange_scalar_rows(v_, 1020);
}

// rank 0에서 출력 디렉터리와 CSV/geometry VTK를 준비한다.
void FVMSolver::initialize_outputs() {
    if (rank_ == 0) {
        std::filesystem::create_directories(opt_.output_dir);
        geom_.write_legacy_vtk(opt_.output_dir + "/" + opt_.output_prefix + "_geometry.vtk");
        drag_csv_.open(opt_.output_dir + "/" + opt_.output_prefix + "_drag.csv");
        if (!drag_csv_) throw std::runtime_error("Cannot open drag CSV in " + opt_.output_dir);
        drag_csv_ << "step,time_s,drag_N_per_m,lift_N_per_m,cd,cl\n";
    }
}

// FVM solver의 실행 설정 요약 출력.
void FVMSolver::print_summary() const {
    if (rank_ != 0) return;
    std::cout << "Case: " << opt_.case_name << "\n"
              << "Solver: finite-volume projection Navier-Stokes + Smagorinsky LES\n"
              << "MPI ranks=" << size_ << ", OpenMP max threads=";
#ifdef _OPENMP
    std::cout << omp_get_max_threads();
#else
    std::cout << 1;
#endif
    std::cout << "\nMesh=" << nx_ << "x" << ny_ << ", dx=" << dx_ << " m, dy=" << dy_ << " m\n"
              << "Fixed dt=" << dt_ << " s, convective CFL=" << opt_.inlet_velocity * dt_ / std::min(dx_, dy_) << "\n"
              << "Re=" << re_ << ", nu=" << nu_ << " m^2/s\n"
              << "Kolmogorov eta≈" << eta_ << " m, dx/eta≈" << std::min(dx_, dy_) / eta_
              << ", LES filter width≈" << opt_.les_filter_width_cells * std::sqrt(dx_ * dy_) << " m\n"
              << "Pressure iterations/step=" << opt_.fvm_pressure_iterations
              << ", pressure tolerance=" << opt_.fvm_pressure_tolerance << "\n"
              << "Geometry=" << geom_.name() << ", bbox=[" << geom_.min_x() << ',' << geom_.min_y()
              << "]..[" << geom_.max_x() << ',' << geom_.max_y() << "]\n" << std::flush;
}

// scalar field halo row 교환.
void FVMSolver::exchange_scalar_rows(std::vector<double>& a, int tag_base) const {
    MPI_Sendrecv(a.data() + cell(local_ny_, 0), nx_, MPI_DOUBLE, upper_rank_, tag_base,
                 a.data() + cell(0, 0), nx_, MPI_DOUBLE, lower_rank_, tag_base,
                 comm_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + cell(1, 0), nx_, MPI_DOUBLE, lower_rank_, tag_base + 1,
                 a.data() + cell(local_ny_ + 1, 0), nx_, MPI_DOUBLE, upper_rank_, tag_base + 1,
                 comm_, MPI_STATUS_IGNORE);
}

// 속도 경계조건: inlet Dirichlet, outlet zero-gradient, solid no-slip,
// bottom moving-ground, top freestream을 적용한다.
void FVMSolver::apply_velocity_bc(std::vector<double>& u, std::vector<double>& v) {
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        // inlet 지정 속도 Dirichlet 조건.
        u[cell(ly, 0)] = opt_.inlet_velocity;
        v[cell(ly, 0)] = 0.0;
        // outlet zero-gradient 조건.
        u[cell(ly, nx_ - 1)] = u[cell(ly, nx_ - 2)];
        v[cell(ly, nx_ - 1)] = v[cell(ly, nx_ - 2)];
        for (int x = 0; x < nx_; ++x) {
            if (is_solid(ly, x)) {
                u[cell(ly, x)] = 0.0;
                v[cell(ly, x)] = 0.0;
            }
        }
    }
    if (y_start_ == 0) {
        const int ly = 1;
#pragma omp parallel for schedule(static)
        for (int x = 0; x < nx_; ++x) {
            if (!is_solid(ly, x)) {
                u[cell(ly, x)] = opt_.inlet_velocity; // 움직이는 지면 조건
                v[cell(ly, x)] = 0.0;
            }
        }
    }
    if (y_start_ + local_ny_ == ny_) {
        const int ly = local_ny_;
#pragma omp parallel for schedule(static)
        for (int x = 0; x < nx_; ++x) {
            if (!is_solid(ly, x)) {
                u[cell(ly, x)] = opt_.inlet_velocity; // 상단 자유류 조건
                v[cell(ly, x)] = 0.0;
            }
        }
    }
}

// 압력 보정 phi의 경계조건. outlet은 reference pressure로 0을 둔다.
void FVMSolver::apply_pressure_bc(std::vector<double>& phi) {
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        phi[cell(ly, 0)] = phi[cell(ly, 1)];       // inlet Neumann 조건
        phi[cell(ly, nx_ - 1)] = 0.0;              // outlet 기준 압력
        for (int x = 0; x < nx_; ++x) {
            if (is_solid(ly, x)) phi[cell(ly, x)] = 0.0;
        }
    }
    if (y_start_ == 0) {
        const int ly = 1;
#pragma omp parallel for schedule(static)
        for (int x = 0; x < nx_; ++x) phi[cell(ly, x)] = phi[cell(ly + 1, x)];
    }
    if (y_start_ + local_ny_ == ny_) {
        const int ly = local_ny_;
#pragma omp parallel for schedule(static)
        for (int x = 0; x < nx_; ++x) phi[cell(ly, x)] = phi[cell(ly - 1, x)];
    }
}

// Smagorinsky LES 점성 계산. molecular nu에 eddy viscosity를 더하되
// 너무 커지는 것을 fvm_nu_eff_max_factor로 제한한다.
void FVMSolver::compute_les_viscosity(const std::vector<double>& u, const std::vector<double>& v) {
    const double delta = opt_.les_filter_width_cells * std::sqrt(dx_ * dy_);
    const double csd2 = opt_.smagorinsky_cs * opt_.smagorinsky_cs * delta * delta;
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            const int c = cell(ly, x);
            if (solid_[c]) {
                nu_eff_[c] = nu_;
                continue;
            }
            const int xm = std::max(0, x - 1);
            const int xp = std::min(nx_ - 1, x + 1);
            const int ym = (global_y(ly) == 0) ? ly : ly - 1;
            const int yp = (global_y(ly) == ny_ - 1) ? ly : ly + 1;
            const double ddx = std::max(dx_, static_cast<double>(xp - xm) * dx_);
            const double ddy = std::max(dy_, static_cast<double>(yp - ym) * dy_);
            const double dudx = (u[cell(ly, xp)] - u[cell(ly, xm)]) / ddx;
            const double dvdx = (v[cell(ly, xp)] - v[cell(ly, xm)]) / ddx;
            const double dudy = (u[cell(yp, x)] - u[cell(ym, x)]) / ddy;
            const double dvdy = (v[cell(yp, x)] - v[cell(ym, x)]) / ddy;
            const double sxx = dudx;
            const double syy = dvdy;
            const double sxy = 0.5 * (dudy + dvdx);
            const double smag = std::sqrt(std::max(0.0, 2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy)));
            nu_eff_[c] = std::min(opt_.fvm_nu_eff_max_factor * nu_, nu_ + csd2 * smag);
        }
    }
    exchange_scalar_rows(nu_eff_, 1040);
}

// explicit predictor에 사용할 RHS 계산.
// 대류항은 face 평균 기반 2차 중심형 flux, 확산항은 variable viscosity 중심차분 형태이다.
void FVMSolver::compute_rhs(const std::vector<double>& u, const std::vector<double>& v,
                            std::vector<double>& ru, std::vector<double>& rv) const {
    const double inv_dx = 1.0 / dx_;
    const double inv_dy = 1.0 / dy_;
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            const int c = cell(ly, x);
            if (solid_[c] || x == 0 || x == nx_ - 1 || global_y(ly) == 0 || global_y(ly) == ny_ - 1) {
                ru[c] = 0.0;
                rv[c] = 0.0;
                continue;
            }
            const int e = cell(ly, x + 1);
            const int w = cell(ly, x - 1);
            const int n = cell(ly + 1, x);
            const int s = cell(ly - 1, x);

            const double ue = 0.5 * (u[c] + u[e]);
            const double uw = 0.5 * (u[w] + u[c]);
            const double un = 0.5 * (u[c] + u[n]);
            const double us = 0.5 * (u[s] + u[c]);
            const double ve = 0.5 * (v[c] + v[e]);
            const double vw = 0.5 * (v[w] + v[c]);
            const double vn = 0.5 * (v[c] + v[n]);
            const double vs = 0.5 * (v[s] + v[c]);

            const double conv_u = ((ue * ue - uw * uw) * inv_dx) + ((vn * un - vs * us) * inv_dy);
            const double conv_v = ((ue * ve - uw * vw) * inv_dx) + ((vn * vn - vs * vs) * inv_dy);

            const double nue = 0.5 * (nu_eff_[c] + nu_eff_[e]);
            const double nuw = 0.5 * (nu_eff_[w] + nu_eff_[c]);
            const double nun = 0.5 * (nu_eff_[c] + nu_eff_[n]);
            const double nus = 0.5 * (nu_eff_[s] + nu_eff_[c]);

            const double diff_u = (nue * (u[e] - u[c]) - nuw * (u[c] - u[w])) / (dx_ * dx_) +
                                  (nun * (u[n] - u[c]) - nus * (u[c] - u[s])) / (dy_ * dy_);
            const double diff_v = (nue * (v[e] - v[c]) - nuw * (v[c] - v[w])) / (dx_ * dx_) +
                                  (nun * (v[n] - v[c]) - nus * (v[c] - v[s])) / (dy_ * dy_);
            ru[c] = -conv_u + diff_u;
            rv[c] = -conv_v + diff_v;
        }
    }
}

// projection method의 pressure Poisson equation을 Jacobi/SOR-like relaxation으로 푼다.
// MPI rank 간 phi halo를 매 반복마다 교환하고 global residual로 조기 종료한다.
double FVMSolver::pressure_poisson() {
    std::fill(phi_.begin(), phi_.end(), 0.0);
    std::fill(phi_new_.begin(), phi_new_.end(), 0.0);
    const double dx2 = dx_ * dx_;
    const double dy2 = dy_ * dy_;
    const double denom = 2.0 * (dx2 + dy2);
    double global_res = 0.0;
    for (int iter = 0; iter < opt_.fvm_pressure_iterations; ++iter) {
        exchange_scalar_rows(phi_, 1060);
        apply_pressure_bc(phi_);
        double local_res = 0.0;
#pragma omp parallel for schedule(static) reduction(max:local_res)
        for (int ly = 1; ly <= local_ny_; ++ly) {
            for (int x = 1; x < nx_ - 1; ++x) {
                const int c = cell(ly, x);
                if (solid_[c] || global_y(ly) == 0 || global_y(ly) == ny_ - 1) {
                    phi_new_[c] = 0.0;
                    continue;
                }
                const int e = cell(ly, x + 1);
                const int w = cell(ly, x - 1);
                const int n = cell(ly + 1, x);
                const int s = cell(ly - 1, x);
                const double pe = solid_[e] ? phi_[c] : phi_[e];
                const double pw = solid_[w] ? phi_[c] : phi_[w];
                const double pn = solid_[n] ? phi_[c] : phi_[n];
                const double ps = solid_[s] ? phi_[c] : phi_[s];
                const double candidate = ((pe + pw) * dy2 + (pn + ps) * dx2 - div_rhs_[c] * dx2 * dy2) / denom;
                const double relaxed = (1.0 - opt_.fvm_pressure_omega) * phi_[c] + opt_.fvm_pressure_omega * candidate;
                local_res = std::max(local_res, std::abs(relaxed - phi_[c]));
                phi_new_[c] = relaxed;
            }
        }
        phi_.swap(phi_new_);
        if ((iter + 1) % 10 == 0 || iter == opt_.fvm_pressure_iterations - 1) {
            MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE, MPI_MAX, comm_);
            if (global_res < opt_.fvm_pressure_tolerance) break;
        }
    }
    apply_pressure_bc(phi_);
    exchange_scalar_rows(phi_, 1080);
    return global_res;
}

// tentative velocity u_star/v_star를 divergence-free에 가깝게 보정한다.
// 보정 pressure phi를 누적해 p_ field도 업데이트한다.
void FVMSolver::project_velocity() {
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 1; x < nx_ - 1; ++x) {
            const int c = cell(ly, x);
            if (solid_[c]) {
                div_rhs_[c] = 0.0;
                continue;
            }
            const int ym = (global_y(ly) == 0) ? ly : ly - 1;
            const int yp = (global_y(ly) == ny_ - 1) ? ly : ly + 1;
            div_rhs_[c] = ((u_star_[cell(ly, x + 1)] - u_star_[cell(ly, x - 1)]) / (2.0 * dx_) +
                           (v_star_[cell(yp, x)] - v_star_[cell(ym, x)]) / (2.0 * dy_)) / dt_;
        }
    }
    pressure_poisson();
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 1; x < nx_ - 1; ++x) {
            const int c = cell(ly, x);
            if (solid_[c]) {
                u_[c] = 0.0;
                v_[c] = 0.0;
                continue;
            }
            const int ym = (global_y(ly) == 0) ? ly : ly - 1;
            const int yp = (global_y(ly) == ny_ - 1) ? ly : ly + 1;
            u_[c] = u_star_[c] - dt_ * (phi_[cell(ly, x + 1)] - phi_[cell(ly, x - 1)]) / (2.0 * dx_);
            v_[c] = v_star_[c] - dt_ * (phi_[cell(yp, x)] - phi_[cell(ym, x)]) / (2.0 * dy_);
            p_[c] += opt_.density * phi_[c];
        }
    }
    apply_velocity_bc(u_, v_);
    exchange_scalar_rows(u_, 1100);
    exchange_scalar_rows(v_, 1120);
    exchange_scalar_rows(p_, 1140);
}

std::vector<double> FVMSolver::local_scalar(const std::vector<double>& a) const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    for (int ly = 1; ly <= local_ny_; ++ly)
        for (int x = 0; x < nx_; ++x) out[static_cast<std::size_t>(ly - 1) * nx_ + x] = a[cell(ly, x)];
    return out;
}

// solid mask를 0/1 scalar로 변환해 VTK에 함께 출력한다.
std::vector<double> FVMSolver::local_solid() const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    for (int ly = 1; ly <= local_ny_; ++ly)
        for (int x = 0; x < nx_; ++x) out[static_cast<std::size_t>(ly - 1) * nx_ + x] = solid_[cell(ly, x)] ? 1.0 : 0.0;
    return out;
}

// wake 구조 확인용 vorticity field. 중앙차분으로 계산한다.
std::vector<double> FVMSolver::local_vorticity() const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(nx_ - 1, x + 1);
            const int ym = (global_y(ly) == 0) ? ly : ly - 1;
            const int yp = (global_y(ly) == ny_ - 1) ? ly : ly + 1;
            const double ddx = std::max(dx_, static_cast<double>(xp - xm) * dx_);
            const double ddy = std::max(dy_, static_cast<double>(yp - ym) * dy_);
            out[static_cast<std::size_t>(ly - 1) * nx_ + x] = solid_[cell(ly, x)] ? 0.0 :
                ((v_[cell(ly, xp)] - v_[cell(ly, xm)]) / ddx - (u_[cell(yp, x)] - u_[cell(ym, x)]) / ddy);
        }
    }
    return out;
}

// LES 포함 유효 점성계수를 출력 field로 사용한다.
std::vector<double> FVMSolver::local_nu_eff() const { return local_scalar(nu_eff_); }

// rank별 local field를 rank 0 전역 field로 gather한다.
std::vector<double> FVMSolver::gather_scalar(const std::vector<double>& local) const {
    const auto counts = counts_for_y(ny_, nx_, size_);
    const auto displs = displs_from_counts(counts);
    std::vector<double> global;
    if (rank_ == 0) global.resize(static_cast<std::size_t>(nx_) * ny_);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                rank_ == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, comm_);
    return global;
}

// ParaView용 VTK 출력.
void FVMSolver::write_vtk(int step) {
    const auto gu = gather_scalar(local_scalar(u_));
    const auto gv = gather_scalar(local_scalar(v_));
    const auto gp = gather_scalar(local_scalar(p_));
    const auto gs = gather_scalar(local_solid());
    const auto gn = gather_scalar(local_nu_eff());
    const auto gw = gather_scalar(local_vorticity());
    if (rank_ == 0) {
        write_structured_points_vtk(step_name(opt_.output_dir, opt_.output_prefix, step), nx_, ny_, dx_, dy_,
                                    {{"pressure_Pa", &gp}, {"solid", &gs}, {"nu_eff_m2_s", &gn}, {"vorticity_s", &gw}},
                                    {{"velocity_m_s", &gu, &gv}});
    }
}

// solid-fluid 인접 face에서 압력력과 점성력을 근사해 drag/lift를 적분한다.
void FVMSolver::write_drag(int step) {
    double local_fx = 0.0;
    double local_fy = 0.0;
#pragma omp parallel for schedule(static) reduction(+:local_fx,local_fy)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 1; x < nx_ - 1; ++x) {
            const int c = cell(ly, x);
            if (solid_[c]) continue;
            const int nbrs[4] = {cell(ly, x + 1), cell(ly, x - 1), cell(ly + 1, x), cell(ly - 1, x)};
            const double nxn[4] = {-1.0, 1.0, 0.0, 0.0}; // 해당 solid 이웃 기준 solid->fluid 법선
            const double nyn[4] = {0.0, 0.0, -1.0, 1.0};
            const double area[4] = {dy_, dy_, dx_, dx_};
            for (int k = 0; k < 4; ++k) {
                if (!solid_[nbrs[k]]) continue;
                const double nxf = nxn[k];
                const double nyf = nyn[k];
                const double dudn = (u_[c] - 0.0) / std::max(dx_, dy_);
                const double dvdn = (v_[c] - 0.0) / std::max(dx_, dy_);
                const double press_fx = -p_[c] * nxf * area[k];
                const double press_fy = -p_[c] * nyf * area[k];
                const double visc_fx = opt_.density * nu_eff_[c] * dudn * area[k];
                const double visc_fy = opt_.density * nu_eff_[c] * dvdn * area[k];
                local_fx += press_fx + visc_fx;
                local_fy += press_fy + visc_fy;
            }
        }
    }
    double global_f[2] = {0.0, 0.0};
    double local_f[2] = {local_fx, local_fy};
    MPI_Reduce(local_f, global_f, 2, MPI_DOUBLE, MPI_SUM, 0, comm_);
    if (rank_ == 0) {
        const double q = 0.5 * opt_.density * opt_.inlet_velocity * opt_.inlet_velocity * ref_height_;
        const double cd = global_f[0] / q;
        const double cl = global_f[1] / q;
        drag_csv_ << step << ',' << step * dt_ << ',' << global_f[0] << ',' << global_f[1] << ',' << cd << ',' << cl << '\n';
        if (step % opt_.log_interval == 0 || step == opt_.steps) {
            std::cout << "step " << step << '/' << opt_.steps << " t=" << step * dt_
                      << " s Cd=" << cd << " Cl=" << cl
                      << " drag_N_per_m=" << global_f[0] << " lift_N_per_m=" << global_f[1] << '\n' << std::flush;
        }
    }
}

// FVM time loop: RK2 predictor로 대류/확산을 적분한 뒤 pressure projection을 수행한다.
void FVMSolver::run() {
    print_summary();
    compute_les_viscosity(u_, v_);
    write_vtk(0);
    write_drag(0);
    for (int step = 1; step <= opt_.steps; ++step) {
        exchange_scalar_rows(u_, 1160);
        exchange_scalar_rows(v_, 1180);
        compute_les_viscosity(u_, v_);
        compute_rhs(u_, v_, rhs_u_, rhs_v_);
#pragma omp parallel for schedule(static)
        for (int ly = 1; ly <= local_ny_; ++ly) {
            for (int x = 0; x < nx_; ++x) {
                const int c = cell(ly, x);
                u_pred_[c] = u_[c] + dt_ * rhs_u_[c];
                v_pred_[c] = v_[c] + dt_ * rhs_v_[c];
            }
        }
        apply_velocity_bc(u_pred_, v_pred_);
        exchange_scalar_rows(u_pred_, 1200);
        exchange_scalar_rows(v_pred_, 1220);
        compute_les_viscosity(u_pred_, v_pred_);
        compute_rhs(u_pred_, v_pred_, rhs_u2_, rhs_v2_);
#pragma omp parallel for schedule(static)
        for (int ly = 1; ly <= local_ny_; ++ly) {
            for (int x = 0; x < nx_; ++x) {
                const int c = cell(ly, x);
                u_star_[c] = u_[c] + 0.5 * dt_ * (rhs_u_[c] + rhs_u2_[c]);
                v_star_[c] = v_[c] + 0.5 * dt_ * (rhs_v_[c] + rhs_v2_[c]);
            }
        }
        apply_velocity_bc(u_star_, v_star_);
        exchange_scalar_rows(u_star_, 1240);
        exchange_scalar_rows(v_star_, 1260);
        project_velocity();
        compute_les_viscosity(u_, v_);
        if (step % opt_.drag_interval == 0 || step == opt_.steps) write_drag(step);
        if (step % opt_.output_interval == 0 || step == opt_.steps) write_vtk(step);
    }
}
