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
// 최종 production 결과를 만든 D2Q9 LBM + Smagorinsky LES solver.
// MPI는 y 방향 sub-domain을 나누고, 각 rank 내부의 cell loop는 OpenMP로 돈다.
// 모든 배열은 halo row 2개(위/아래)를 포함한 1차원 vector로 저장한다.
class LBMSolver {
public:
    LBMSolver(const Options& opt, const Geometry2D& geom, MPI_Comm comm);
    void run();

private:
    // D2Q9 격자 속도, 반대 방향 index, 가중치.
    static constexpr int Q = 9;
    static constexpr std::array<int, Q> cx{{0, 1, 0, -1, 0, 1, -1, -1, 1}};
    static constexpr std::array<int, Q> cy{{0, 0, 1, 0, -1, 1, 1, -1, -1}};
    static constexpr std::array<int, Q> opp{{0, 3, 4, 1, 2, 7, 8, 5, 6}};
    static constexpr std::array<double, Q> wt{{4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                                               1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}};

    const Options opt_;       // 실행 중 옵션은 변하지 않도록 복사 보관한다.
    const Geometry2D& geom_;  // geometry는 solver 외부에서 만든 solid 정의를 참조한다.
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int size_ = 1;

    // 전역 mesh와 현재 MPI rank가 맡은 y 구간.
    int nx_ = 0;
    int ny_ = 0;
    int local_ny_ = 0;
    int y_start_ = 0;
    int lower_rank_ = MPI_PROC_NULL;
    int upper_rank_ = MPI_PROC_NULL;

    // 물리 단위와 lattice 단위 변환에 필요한 스케일.
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

    // f_: 현재 분포함수, post_: collision 후 분포함수, f_next_: streaming 결과.
    // rho_/ux_/uy_는 lattice 단위 macro field이고, tau_eff_는 LES 포함 relaxation time이다.
    std::vector<double> f_;
    std::vector<double> post_;
    std::vector<double> f_next_;
    std::vector<double> rho_;
    std::vector<double> ux_;
    std::vector<double> uy_;
    std::vector<double> tau_eff_;
    std::vector<unsigned char> solid_;

    // rank 0만 열어 쓰는 항력/양력 시간 이력 파일.
    std::ofstream drag_csv_;

    // 1차원 배열 index helper. ly=1..local_ny_가 실제 interior, 0/local_ny_+1은 halo.
    [[nodiscard]] int cell(int ly, int x) const { return ly * nx_ + x; }
    [[nodiscard]] int idx(int ly, int x, int q) const { return (cell(ly, x) * Q) + q; }
    [[nodiscard]] int global_y(int ly) const { return y_start_ + (ly - 1); }
    [[nodiscard]] double phys_x(int x) const { return static_cast<double>(x) * dx_; }
    [[nodiscard]] double phys_y_from_global(int gy) const { return static_cast<double>(gy) * dy_; }
    [[nodiscard]] bool in_global_y(int gy) const { return gy >= 0 && gy < ny_; }
    [[nodiscard]] bool is_solid(int ly, int x) const { return solid_[cell(ly, x)] != 0; }

    void decompose();
    void initialize_fields();
    void initialize_outputs();
    void print_summary(const std::vector<std::string>& runtime_warnings) const;

    // LBM 핵심 단계: equilibrium -> macro 계산 -> LES tau -> collision -> streaming/bounce-back -> boundary.
    double equilibrium(int q, double rho, double ux, double uy) const;
    void compute_macros();
    void exchange_scalar_rows(std::vector<double>& a, int tag_base);
    void exchange_dist_rows(std::vector<double>& a, int tag_base);
    void compute_les_tau();
    void collide();
    std::array<double, 2> stream_and_bounce();
    void apply_boundaries();

    // 출력용 gather/field 생성 함수. rank별 local field를 rank 0에서 전역 VTK field로 합친다.
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
