#include "lbm_solver.hpp"

#include "vtk_writer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// 이 파일은 최종 결과 생성에 사용한 production solver이다.
// 계산 순서는 macro field 계산 -> LES relaxation time 계산 -> collision ->
// streaming/bounce-back -> boundary condition -> output/drag 기록으로 구성된다.
// MPI 통신은 y 방향 halo row 교환에만 사용하고, 격자 내부 반복은 OpenMP로 병렬화한다.

namespace {

// MPI_Gatherv로 rank별 local row를 모을 때 필요한 수신 count 계산.
std::vector<int> counts_for_y(int ny, int nx, int size) {
    std::vector<int> counts(size, 0);
    const int base = ny / size;
    const int rem = ny % size;
    for (int r = 0; r < size; ++r) counts[r] = (base + (r < rem ? 1 : 0)) * nx;
    return counts;
}

// counts 배열에서 displacement 배열을 만든다.
std::vector<int> displs_from_counts(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size(), 0);
    for (std::size_t i = 1; i < counts.size(); ++i) displs[i] = displs[i - 1] + counts[i - 1];
    return displs;
}

// step 번호가 포함된 VTK 파일 이름을 일관되게 만든다.
std::string step_name(const std::string& dir, const std::string& prefix, int step) {
    std::ostringstream path;
    path << dir << '/' << prefix << '_' << std::setw(7) << std::setfill('0') << step << ".vtk";
    return path.str();
}

} // 익명 namespace

// constructor에서 모든 물리/lattice 스케일을 계산하고 배열을 할당한다.
// dt는 CFL로 여기서 한 번만 계산되며 run 중에는 절대 변경하지 않는다.
LBMSolver::LBMSolver(const Options& opt, const Geometry2D& geom, MPI_Comm comm)
    : opt_(opt), geom_(geom), comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    nx_ = opt_.nx;
    ny_ = opt_.ny;
    if (ny_ < size_) {
        throw std::runtime_error("Number of MPI ranks must not exceed mesh.ny");
    }
    decompose();

    // 물리 격자와 lattice 단위 변환.
    dx_ = opt_.domain_length / static_cast<double>(nx_ - 1);
    dy_ = opt_.domain_height / static_cast<double>(ny_ - 1);
    dt_ = opt_.cfl * dx_ / opt_.inlet_velocity;
    u_lu_ = opt_.inlet_velocity * dt_ / dx_;

    // mu가 주어지면 nu=mu/rho, 아니면 Reynolds 수로부터 nu=U*L/Re 계산.
    if (opt_.dynamic_viscosity > 0.0) {
        nu_phys_ = opt_.dynamic_viscosity / opt_.density;
        re_ = opt_.inlet_velocity * opt_.car_length / nu_phys_;
    } else {
        re_ = opt_.reynolds;
        nu_phys_ = opt_.inlet_velocity * opt_.car_length / re_;
    }
    nu_lu_ = nu_phys_ * dt_ / (dx_ * dx_);
    tau0_ = 0.5 + 3.0 * nu_lu_;
    kolmogorov_eta_ = opt_.car_length * std::pow(std::max(re_, 1.0), -0.75);
    ref_height_ = geom_.height() > 0.0 ? geom_.height() : opt_.car_height;

    // halo 2줄을 포함한 local 배열 할당.
    const std::size_t cells_with_halo = static_cast<std::size_t>(local_ny_ + 2) * static_cast<std::size_t>(nx_);
    f_.assign(cells_with_halo * Q, 0.0);
    post_.assign(cells_with_halo * Q, 0.0);
    f_next_.assign(cells_with_halo * Q, 0.0);
    rho_.assign(cells_with_halo, 1.0);
    ux_.assign(cells_with_halo, 0.0);
    uy_.assign(cells_with_halo, 0.0);
    tau_eff_.assign(cells_with_halo, tau0_);
    solid_.assign(cells_with_halo, 0);

    initialize_fields();
    initialize_outputs();
}

// y 방향으로 행을 나누는 1D domain decomposition.
// 나머지 row는 앞 rank부터 하나씩 더 받아 load balance를 맞춘다.
void LBMSolver::decompose() {
    const int base = ny_ / size_;
    const int rem = ny_ % size_;
    local_ny_ = base + (rank_ < rem ? 1 : 0);
    y_start_ = rank_ * base + std::min(rank_, rem);
    lower_rank_ = (y_start_ > 0) ? rank_ - 1 : MPI_PROC_NULL;
    upper_rank_ = (y_start_ + local_ny_ < ny_) ? rank_ + 1 : MPI_PROC_NULL;
}

// D2Q9 equilibrium distribution. LBM collision 단계의 기준 분포이다.
double LBMSolver::equilibrium(int q, double rho, double ux, double uy) const {
    const double cu = 3.0 * (static_cast<double>(cx[q]) * ux + static_cast<double>(cy[q]) * uy);
    const double uu = ux * ux + uy * uy;
    return wt[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * uu);
}

// 초기 조건: 전체 유동을 inlet 속도 근처로 놓고 작은 perturbation을 더한다.
// 차량 내부 solid cell은 속도 0, 분포함수는 정지 equilibrium으로 초기화한다.
void LBMSolver::initialize_fields() {
    for (int ly = 0; ly < local_ny_ + 2; ++ly) {
        const int gy = global_y(ly);
        const bool valid_y = in_global_y(gy);
        const double y = valid_y ? phys_y_from_global(gy) : -1.0;
        for (int x = 0; x < nx_; ++x) {
            const double px = phys_x(x);
            const bool solid = valid_y && geom_.contains(px, y);
            solid_[cell(ly, x)] = solid ? 1 : 0;
            const double phase = 2.0 * 3.14159265358979323846 *
                                 (static_cast<double>(x) / std::max(1, nx_ - 1) +
                                  static_cast<double>(std::max(gy, 0)) / std::max(1, ny_ - 1));
            double ux = solid ? 0.0 : u_lu_ * (1.0 + opt_.perturbation * std::sin(phase));
            double uy = solid ? 0.0 : u_lu_ * opt_.perturbation * 0.25 * std::cos(phase);
            if (x == 0 || x == nx_ - 1) {
                ux = solid ? 0.0 : u_lu_;
                uy = 0.0;
            }
            rho_[cell(ly, x)] = 1.0;
            ux_[cell(ly, x)] = ux;
            uy_[cell(ly, x)] = uy;
            tau_eff_[cell(ly, x)] = tau0_;
            for (int q = 0; q < Q; ++q) {
                f_[idx(ly, x, q)] = equilibrium(q, 1.0, ux, uy);
            }
        }
    }
}

// rank 0이 출력 디렉터리, geometry VTK, drag CSV를 준비한다.
void LBMSolver::initialize_outputs() {
    if (rank_ == 0) {
        std::filesystem::create_directories(opt_.output_dir);
        geom_.write_legacy_vtk(opt_.output_dir + "/" + opt_.output_prefix + "_geometry.vtk");
        drag_csv_.open(opt_.output_dir + "/" + opt_.output_prefix + "_drag.csv");
        if (!drag_csv_) throw std::runtime_error("Cannot open drag CSV in " + opt_.output_dir);
        drag_csv_ << "step,time_s,drag_lu,lift_lu,cd,cl,drag_N_per_m,lift_N_per_m\n";
    }
}

// 실행 시작 시 사용자가 확인해야 할 물리/수치 스케일을 rank 0에 출력한다.
void LBMSolver::print_summary(const std::vector<std::string>& runtime_warnings) const {
    if (rank_ != 0) return;
    std::cout << "Case: " << opt_.case_name << "\n"
              << "MPI ranks=" << size_ << ", OpenMP max threads=";
#ifdef _OPENMP
    std::cout << omp_get_max_threads();
#else
    std::cout << 1;
#endif
    std::cout << "\nMesh=" << nx_ << "x" << ny_ << ", dx=" << dx_ << " m, dy=" << dy_ << " m\n"
              << "Fixed dt=" << dt_ << " s, CFL/lattice U=" << u_lu_ << "\n"
              << "Re=" << re_ << ", nu_phys=" << nu_phys_ << " m^2/s, nu_lu=" << nu_lu_
              << ", tau0=" << tau0_ << "\n"
              << "Kolmogorov eta≈" << kolmogorov_eta_ << " m, dx/eta≈" << dx_ / kolmogorov_eta_
              << ", LES filter width≈" << opt_.les_filter_width_cells * dx_ << " m\n"
              << "Geometry=" << geom_.name() << ", bbox=[" << geom_.min_x() << ',' << geom_.min_y()
              << "]..[" << geom_.max_x() << ',' << geom_.max_y() << "]\n";
    for (const auto& w : runtime_warnings) std::cout << "Warning: " << w << "\n";
    if (tau0_ <= opt_.tau_min + 1.0e-12) {
        std::cout << "Warning: molecular tau is close to 0.5; high-Re LBM runs need finer meshes, lower CFL, "
                     "or LES eddy viscosity to remain stable.\n";
    }
    std::cout << std::flush;
}

// scalar field halo row 교환. ux/uy 같은 cell field에 사용한다.
void LBMSolver::exchange_scalar_rows(std::vector<double>& a, int tag_base) {
    const int row = nx_;
    MPI_Sendrecv(a.data() + cell(local_ny_, 0), row, MPI_DOUBLE, upper_rank_, tag_base,
                 a.data() + cell(0, 0), row, MPI_DOUBLE, lower_rank_, tag_base,
                 comm_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + cell(1, 0), row, MPI_DOUBLE, lower_rank_, tag_base + 1,
                 a.data() + cell(local_ny_ + 1, 0), row, MPI_DOUBLE, upper_rank_, tag_base + 1,
                 comm_, MPI_STATUS_IGNORE);
}

// 분포함수 f/post 배열은 방향 Q까지 포함하므로 row 크기가 nx*Q이다.
void LBMSolver::exchange_dist_rows(std::vector<double>& a, int tag_base) {
    const int row = nx_ * Q;
    MPI_Sendrecv(a.data() + idx(local_ny_, 0, 0), row, MPI_DOUBLE, upper_rank_, tag_base,
                 a.data() + idx(0, 0, 0), row, MPI_DOUBLE, lower_rank_, tag_base,
                 comm_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + idx(1, 0, 0), row, MPI_DOUBLE, lower_rank_, tag_base + 1,
                 a.data() + idx(local_ny_ + 1, 0, 0), row, MPI_DOUBLE, upper_rank_, tag_base + 1,
                 comm_, MPI_STATUS_IGNORE);
}

// 분포함수 합으로 density와 velocity를 계산한다.
// solid cell은 안정성을 위해 rho=1, velocity=0으로 고정한다.
void LBMSolver::compute_macros() {
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            const int c = cell(ly, x);
            if (solid_[c]) {
                rho_[c] = 1.0;
                ux_[c] = 0.0;
                uy_[c] = 0.0;
                continue;
            }
            double rho = 0.0;
            double jx = 0.0;
            double jy = 0.0;
            for (int q = 0; q < Q; ++q) {
                const double fq = f_[idx(ly, x, q)];
                rho += fq;
                jx += fq * static_cast<double>(cx[q]);
                jy += fq * static_cast<double>(cy[q]);
            }
            if (!std::isfinite(rho) || rho <= 1.0e-12) rho = 1.0;
            rho_[c] = rho;
            ux_[c] = jx / rho;
            uy_[c] = jy / rho;
        }
    }
    exchange_scalar_rows(ux_, 300);
    exchange_scalar_rows(uy_, 320);
}

// Smagorinsky LES 모델: local strain-rate magnitude로 eddy viscosity를 더하고
// relaxation time tau_eff를 tau_min/tau_max 범위 안으로 제한한다.
void LBMSolver::compute_les_tau() {
    const double cs2 = opt_.smagorinsky_cs * opt_.smagorinsky_cs;
    const double delta2 = opt_.les_filter_width_cells * opt_.les_filter_width_cells;
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            const int c = cell(ly, x);
            if (solid_[c]) {
                tau_eff_[c] = tau0_;
                continue;
            }
            const int xm = std::max(0, x - 1);
            const int xp = std::min(nx_ - 1, x + 1);
            const int lym = (global_y(ly) == 0) ? ly : ly - 1;
            const int lyp = (global_y(ly) == ny_ - 1) ? ly : ly + 1;

            auto ux_at = [&](int yy, int xx) {
                return solid_[cell(yy, xx)] ? ux_[c] : ux_[cell(yy, xx)];
            };
            auto uy_at = [&](int yy, int xx) {
                return solid_[cell(yy, xx)] ? uy_[c] : uy_[cell(yy, xx)];
            };

            const double ddx = (xp == xm) ? 1.0 : static_cast<double>(xp - xm);
            const double ddy = (lyp == lym) ? 1.0 : static_cast<double>(lyp - lym);
            const double duxdx = (ux_at(ly, xp) - ux_at(ly, xm)) / ddx;
            const double duydx = (uy_at(ly, xp) - uy_at(ly, xm)) / ddx;
            const double duxdy = (ux_at(lyp, x) - ux_at(lym, x)) / ddy;
            const double duydy = (uy_at(lyp, x) - uy_at(lym, x)) / ddy;

            const double sxx = duxdx;
            const double syy = duydy;
            const double sxy = 0.5 * (duxdy + duydx);
            const double smag = std::sqrt(std::max(0.0, 2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy)));
            const double nu_t = cs2 * delta2 * smag;
            double tau = 0.5 + 3.0 * (nu_lu_ + nu_t);
            tau = std::min(std::max(tau, opt_.tau_min), opt_.tau_max);
            tau_eff_[c] = tau;
        }
    }
}

// BGK collision 단계. 각 방향 분포함수를 local equilibrium 쪽으로 relaxation한다.
void LBMSolver::collide() {
#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            const int c = cell(ly, x);
            if (solid_[c]) {
                for (int q = 0; q < Q; ++q) post_[idx(ly, x, q)] = equilibrium(q, 1.0, 0.0, 0.0);
                continue;
            }
            const double omega = 1.0 / tau_eff_[c];
            const double rho = rho_[c];
            const double ux = ux_[c];
            const double uy = uy_[c];
            for (int q = 0; q < Q; ++q) {
                const double feq = equilibrium(q, rho, ux, uy);
                post_[idx(ly, x, q)] = f_[idx(ly, x, q)] - omega * (f_[idx(ly, x, q)] - feq);
            }
        }
    }
    exchange_dist_rows(post_, 400);
}

// streaming과 immersed solid bounce-back을 동시에 수행한다.
// solid 이웃에서 반사되는 분포의 운동량 변화량을 누적해 drag/lift 계산에 사용한다.
std::array<double, 2> LBMSolver::stream_and_bounce() {
    double fx = 0.0;
    double fy = 0.0;
#pragma omp parallel for schedule(static) reduction(+:fx,fy)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            const int c = cell(ly, x);
            if (solid_[c]) {
                for (int q = 0; q < Q; ++q) f_next_[idx(ly, x, q)] = equilibrium(q, 1.0, 0.0, 0.0);
                continue;
            }
            for (int q = 0; q < Q; ++q) {
                const int sx = x - cx[q];
                const int sly = ly - cy[q];
                if (sx < 0 || sx >= nx_ || sly < 0 || sly > local_ny_ + 1) {
                    f_next_[idx(ly, x, q)] = post_[idx(ly, x, q)];
                    continue;
                }
                if (solid_[cell(sly, sx)]) {
                    const int qo = opp[q];
                    const double bounced = post_[idx(ly, x, qo)];
                    f_next_[idx(ly, x, q)] = bounced;
                    fx += 2.0 * static_cast<double>(cx[qo]) * bounced;
                    fy += 2.0 * static_cast<double>(cy[qo]) * bounced;
                } else {
                    f_next_[idx(ly, x, q)] = post_[idx(sly, sx, q)];
                }
            }
        }
    }
    f_.swap(f_next_);
    return {fx, fy};
}

// 외부 경계조건 적용.
// inlet/top/bottom은 prescribed velocity 계열, outlet은 긴 wake 안정성을 위해 zero-gradient이다.
void LBMSolver::apply_boundaries() {
    const double ux_bc = u_lu_;
    const double uy_bc = 0.0;

#pragma omp parallel for schedule(static)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        // 서쪽 inlet: 지정 속도 Zou-He 조건.
        int x = 0;
        if (!is_solid(ly, x)) {
            const double f0 = f_[idx(ly, x, 0)], f2 = f_[idx(ly, x, 2)], f4 = f_[idx(ly, x, 4)];
            const double f3 = f_[idx(ly, x, 3)], f6 = f_[idx(ly, x, 6)], f7 = f_[idx(ly, x, 7)];
            const double rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux_bc);
            f_[idx(ly, x, 1)] = f3 + (2.0 / 3.0) * rho * ux_bc;
            f_[idx(ly, x, 5)] = f7 + (1.0 / 6.0) * rho * ux_bc + 0.5 * rho * uy_bc - 0.5 * (f2 - f4);
            f_[idx(ly, x, 8)] = f6 + (1.0 / 6.0) * rho * ux_bc - 0.5 * rho * uy_bc + 0.5 * (f2 - f4);
        }

        // 동쪽 outlet: zero-gradient outflow.
        // fixed-pressure Zou-He outlet은 짧은 run에서는 가능했지만 긴 wake를 반사해
        // 장시간 적분에서 밀도 음수/불안정을 만들 수 있어 현재 조건으로 바꿨다.
        x = nx_ - 1;
        if (!is_solid(ly, x)) {
            for (int q = 0; q < Q; ++q) f_[idx(ly, x, q)] = f_[idx(ly, x - 1, q)];
        }
    }

    // 아래쪽 ground/freestream 경계.
    if (y_start_ == 0) {
        const int ly = 1;
#pragma omp parallel for schedule(static)
        for (int x = 0; x < nx_; ++x) {
            if (is_solid(ly, x)) continue;
            const double f0 = f_[idx(ly, x, 0)], f1 = f_[idx(ly, x, 1)], f3 = f_[idx(ly, x, 3)];
            const double f4 = f_[idx(ly, x, 4)], f7 = f_[idx(ly, x, 7)], f8 = f_[idx(ly, x, 8)];
            const double rho = (f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)) / (1.0 - uy_bc);
            f_[idx(ly, x, 2)] = f4 + (2.0 / 3.0) * rho * uy_bc;
            f_[idx(ly, x, 5)] = f7 + (1.0 / 6.0) * rho * uy_bc + 0.5 * rho * ux_bc - 0.5 * (f1 - f3);
            f_[idx(ly, x, 6)] = f8 + (1.0 / 6.0) * rho * uy_bc - 0.5 * rho * ux_bc + 0.5 * (f1 - f3);
        }
    }

    // 위쪽 freestream 경계.
    if (y_start_ + local_ny_ == ny_) {
        const int ly = local_ny_;
#pragma omp parallel for schedule(static)
        for (int x = 0; x < nx_; ++x) {
            if (is_solid(ly, x)) continue;
            const double f0 = f_[idx(ly, x, 0)], f1 = f_[idx(ly, x, 1)], f3 = f_[idx(ly, x, 3)];
            const double f2 = f_[idx(ly, x, 2)], f5 = f_[idx(ly, x, 5)], f6 = f_[idx(ly, x, 6)];
            const double rho = (f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)) / (1.0 + uy_bc);
            f_[idx(ly, x, 4)] = f2 - (2.0 / 3.0) * rho * uy_bc;
            f_[idx(ly, x, 7)] = f5 - (1.0 / 6.0) * rho * uy_bc - 0.5 * rho * ux_bc + 0.5 * (f1 - f3);
            f_[idx(ly, x, 8)] = f6 - (1.0 / 6.0) * rho * uy_bc + 0.5 * rho * ux_bc - 0.5 * (f1 - f3);
        }
    }

    // corner는 여러 Zou-He 조건이 겹쳐 over-constraint가 생길 수 있어 equilibrium으로 재설정한다.
    if (y_start_ == 0) {
        const int ly = 1;
        for (int x : {0, nx_ - 1}) {
            if (!is_solid(ly, x)) for (int q = 0; q < Q; ++q) f_[idx(ly, x, q)] = equilibrium(q, 1.0, ux_bc, 0.0);
        }
    }
    if (y_start_ + local_ny_ == ny_) {
        const int ly = local_ny_;
        for (int x : {0, nx_ - 1}) {
            if (!is_solid(ly, x)) for (int q = 0; q < Q; ++q) f_[idx(ly, x, q)] = equilibrium(q, 1.0, ux_bc, 0.0);
        }
    }
}

// density field는 lattice 단위 그대로 출력한다.
std::vector<double> LBMSolver::make_local_scalar_rho() const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    for (int ly = 1; ly <= local_ny_; ++ly)
        for (int x = 0; x < nx_; ++x) out[static_cast<std::size_t>(ly - 1) * nx_ + x] = rho_[cell(ly, x)];
    return out;
}

// 출력은 물리 단위가 이해하기 쉬우므로 ux lattice 값을 [m/s]로 변환한다.
std::vector<double> LBMSolver::make_local_scalar_ux_phys() const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    const double scale = dx_ / dt_;
    for (int ly = 1; ly <= local_ny_; ++ly)
        for (int x = 0; x < nx_; ++x) out[static_cast<std::size_t>(ly - 1) * nx_ + x] = ux_[cell(ly, x)] * scale;
    return out;
}

// uy도 ux와 같은 scale(dx/dt)로 물리 단위 변환한다.
std::vector<double> LBMSolver::make_local_scalar_uy_phys() const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    const double scale = dx_ / dt_;
    for (int ly = 1; ly <= local_ny_; ++ly)
        for (int x = 0; x < nx_; ++x) out[static_cast<std::size_t>(ly - 1) * nx_ + x] = uy_[cell(ly, x)] * scale;
    return out;
}

// solid mask를 0/1 scalar field로 출력해 ParaView에서 형상 위치를 확인한다.
std::vector<double> LBMSolver::make_local_scalar_solid() const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    for (int ly = 1; ly <= local_ny_; ++ly)
        for (int x = 0; x < nx_; ++x) out[static_cast<std::size_t>(ly - 1) * nx_ + x] = solid_[cell(ly, x)] ? 1.0 : 0.0;
    return out;
}

// LES 포함 유효 점성계수를 lattice 단위로 출력한다.
std::vector<double> LBMSolver::make_local_scalar_nu_eff() const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    for (int ly = 1; ly <= local_ny_; ++ly)
        for (int x = 0; x < nx_; ++x) {
            const double nu_eff = std::max(0.0, (tau_eff_[cell(ly, x)] - 0.5) / 3.0);
            out[static_cast<std::size_t>(ly - 1) * nx_ + x] = nu_eff;
        }
    return out;
}

// vorticity는 시각화에서 wake와 separation을 보기 위한 diagnostic field이다.
// 중앙차분을 쓰고, solid cell은 0으로 둔다.
std::vector<double> LBMSolver::make_local_scalar_vorticity() const {
    std::vector<double> out(static_cast<std::size_t>(local_ny_) * nx_);
    const double vel_scale = dx_ / dt_;
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(nx_ - 1, x + 1);
            const int lym = (global_y(ly) == 0) ? ly : ly - 1;
            const int lyp = (global_y(ly) == ny_ - 1) ? ly : ly + 1;
            const double ddx = (xp == xm) ? dx_ : (static_cast<double>(xp - xm) * dx_);
            const double ddy = (lyp == lym) ? dy_ : (static_cast<double>(lyp - lym) * dy_);
            const double duy_dx = (uy_[cell(ly, xp)] - uy_[cell(ly, xm)]) * vel_scale / ddx;
            const double dux_dy = (ux_[cell(lyp, x)] - ux_[cell(lym, x)]) * vel_scale / ddy;
            out[static_cast<std::size_t>(ly - 1) * nx_ + x] = solid_[cell(ly, x)] ? 0.0 : (duy_dx - dux_dy);
        }
    }
    return out;
}

// local field를 rank 0 전역 field로 모은다. VTK write 시점에만 호출되어
// 매 timestep gather 비용이 발생하지 않도록 했다.
std::vector<double> LBMSolver::gather_scalar(const std::vector<double>& local) const {
    const auto counts = counts_for_y(ny_, nx_, size_);
    const auto displs = displs_from_counts(counts);
    std::vector<double> global;
    if (rank_ == 0) global.resize(static_cast<std::size_t>(nx_) * ny_);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                rank_ == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, comm_);
    return global;
}

// ParaView용 VTK 파일 출력. 출력 직전에 macro/LES field를 최신화한 뒤 gather한다.
void LBMSolver::write_vtk(int step) {
    compute_macros();
    compute_les_tau();
    const auto rho = gather_scalar(make_local_scalar_rho());
    const auto ux = gather_scalar(make_local_scalar_ux_phys());
    const auto uy = gather_scalar(make_local_scalar_uy_phys());
    const auto solid = gather_scalar(make_local_scalar_solid());
    const auto nu_eff = gather_scalar(make_local_scalar_nu_eff());
    const auto vort = gather_scalar(make_local_scalar_vorticity());
    if (rank_ == 0) {
        write_structured_points_vtk(step_name(opt_.output_dir, opt_.output_prefix, step), nx_, ny_, dx_, dy_,
                                    {{"rho_lu", &rho}, {"solid", &solid}, {"nu_eff_lu", &nu_eff}, {"vorticity_s", &vort}},
                                    {{"velocity_m_s", &ux, &uy}});
    }
}

// 각 rank가 계산한 momentum-exchange force를 rank 0에서 합산해 CSV에 저장한다.
// Cd/Cl은 2D 단위폭 해석 기준으로 reference height를 사용한다.
void LBMSolver::write_drag(int step, double local_fx, double local_fy) {
    double global_f[2] = {0.0, 0.0};
    double local_f[2] = {local_fx, local_fy};
    MPI_Reduce(local_f, global_f, 2, MPI_DOUBLE, MPI_SUM, 0, comm_);
    if (rank_ == 0) {
        const double h_lu = std::max(ref_height_ / dx_, 1.0);
        const double q_lu = 0.5 * 1.0 * u_lu_ * u_lu_ * h_lu;
        const double cd = global_f[0] / q_lu;
        const double cl = global_f[1] / q_lu;
        const double q_phys = 0.5 * opt_.density * opt_.inlet_velocity * opt_.inlet_velocity * ref_height_;
        const double drag_per_m = cd * q_phys;
        const double lift_per_m = cl * q_phys;
        drag_csv_ << step << ',' << (static_cast<double>(step) * dt_) << ','
                  << global_f[0] << ',' << global_f[1] << ',' << cd << ',' << cl << ','
                  << drag_per_m << ',' << lift_per_m << '\n';
        if (step % opt_.log_interval == 0) log_step(step, global_f[0], global_f[1]);
    }
}

// 긴 run 중 진행 상황과 force coefficient를 확인하기 위한 console log.
void LBMSolver::log_step(int step, double global_fx, double global_fy) const {
    const double h_lu = std::max(ref_height_ / dx_, 1.0);
    const double q_lu = 0.5 * u_lu_ * u_lu_ * h_lu;
    const double cd = global_fx / q_lu;
    const double cl = global_fy / q_lu;
    std::cout << "step " << step << "/" << opt_.steps
              << " t=" << step * dt_ << " s Cd=" << cd << " Cl=" << cl
              << " drag_lu=" << global_fx << " lift_lu=" << global_fy << '\n' << std::flush;
}

// NaN/Inf가 생기면 즉시 모든 rank가 감지하고 예외를 던져 잘못된 결과 저장을 막는다.
void LBMSolver::ensure_finite(int step) const {
    int local_bad = 0;
#pragma omp parallel for schedule(static) reduction(max:local_bad)
    for (int ly = 1; ly <= local_ny_; ++ly) {
        for (int x = 0; x < nx_; ++x) {
            for (int q = 0; q < Q; ++q) {
                if (!std::isfinite(f_[idx(ly, x, q)])) local_bad = 1;
            }
            const int c = cell(ly, x);
            if (!std::isfinite(rho_[c]) || !std::isfinite(ux_[c]) ||
                !std::isfinite(uy_[c]) || !std::isfinite(tau_eff_[c])) {
                local_bad = 1;
            }
        }
    }
    int global_bad = 0;
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, comm_);
    if (global_bad != 0) {
        std::ostringstream msg;
        msg << "Non-finite LBM field detected at step " << step
            << " (t=" << static_cast<double>(step) * dt_ << " s)";
        throw std::runtime_error(msg.str());
    }
}

// 전체 time integration loop. dt는 constructor에서 정한 고정값만 사용한다.
// 출력/drag/finite check는 interval 기반으로 수행한다.
void LBMSolver::run() {
    std::vector<std::string> runtime_warnings;
    if (tau0_ < 0.505) {
        runtime_warnings.push_back("tau0 < 0.505; this is a high-Re/coarse-grid regime for LBM. Use finer mesh or lower Reynolds for stable validation runs.");
    }
    if (dx_ / kolmogorov_eta_ > opt_.les_delta_to_eta * 2.0) {
        std::ostringstream msg;
        msg << "dx/eta=" << dx_ / kolmogorov_eta_ << " exceeds roughly 2x mesh.les_delta_to_eta="
            << opt_.les_delta_to_eta << "; LES resolution is under-resolved for the requested Re.";
        runtime_warnings.push_back(msg.str());
    }
    print_summary(runtime_warnings);

    write_vtk(0);
    double last_fx = 0.0;
    double last_fy = 0.0;
    write_drag(0, 0.0, 0.0);

    for (int step = 1; step <= opt_.steps; ++step) {
        compute_macros();
        compute_les_tau();
        collide();
        const auto force = stream_and_bounce();
        apply_boundaries();
        last_fx = force[0];
        last_fy = force[1];
        if (step % opt_.drag_interval == 0 || step == opt_.steps) ensure_finite(step);
        if (step % opt_.drag_interval == 0 || step == opt_.steps) write_drag(step, last_fx, last_fy);
        if (step % opt_.output_interval == 0 || step == opt_.steps) write_vtk(step);
    }
}
