#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
// 전체 solver가 사용하는 모든 입력 파라미터를 한 곳에 모은 구조체이다.
// INI 파일과 --set CLI override는 결국 이 구조체의 필드로 변환된다.
// 단위가 있는 물리량은 주석에 SI 단위를 명시해서 config 파일을 읽을 때
// 어떤 값이 실제 물리 스케일을 의미하는지 바로 확인할 수 있게 했다.
struct Options {
    std::string case_name = "f1_demo";
    std::string solver_method = "lbm"; // "lbm"은 최종 검증 경로, "fvm"은 실험용 FVM 경로

    // 계산 영역: 자동차를 옆에서 본 2차원 풍동 도메인 크기.
    double domain_length = 18.0;   // [m]
    double domain_height = 4.5;    // [m]

    // 격자 및 LES 해상도 옵션. auto_mesh_from_kolmogorov=true이면
    // Kolmogorov 길이 estimate를 기준으로 nx, ny를 자동 산정한다.
    int nx = 361;
    int ny = 91;
    bool auto_mesh_from_kolmogorov = false;
    double les_delta_to_eta = 20.0;
    std::uint64_t max_cells = 20000000ULL;
    bool allow_large_mesh = false;

    // 유동 물성 및 시간 간격 결정에 쓰이는 값. dynamic_viscosity가 0 이하이면
    // Reynolds 수로부터 kinematic viscosity를 계산한다.
    double inlet_velocity = 30.0;  // [m/s]
    double density = 1.225;        // [kg/m^3]
    double dynamic_viscosity = 0.0;// [Pa s]; 0 이하이면 Reynolds 수로부터 계산
    double reynolds = 2000.0;      // geometry.car_length 기준 Reynolds 수
    double cfl = 0.15;             // 고정 CFL 목표값; 계산 중 dt는 바꾸지 않음
    double perturbation = 0.001;   // 초기 속도장에 더하는 작은 교란 진폭

    // Smagorinsky LES 및 LBM relaxation 안정화 한계.
    double smagorinsky_cs = 0.30;
    double les_filter_width_cells = 1.0;
    double tau_min = 0.5001;
    double tau_max = 2.5;

    // FVM solver 전용 압력 Poisson/점성 제한 옵션.
    int fvm_pressure_iterations = 120;
    double fvm_pressure_tolerance = 1.0e-7;
    double fvm_pressure_omega = 0.85;
    double fvm_nu_eff_max_factor = 200.0;

    // 시간 적분 및 출력 주기. dt는 CFL에서 한 번만 계산되고 run 중 고정된다.
    int steps = 240000;          // 기본 dx=1/60 m, U=30 m/s, CFL=0.15에서 20초
    int output_interval = 1200;  // 기본 고정 dt에서 VTK 저장 간격 0.1초
    int drag_interval = 1200;
    int log_interval = 1200;

    // 형상 선택/배치. preset 또는 geometry_file 중 하나로 차량 형상을 만든다.
    std::string geometry_preset = "generic_f1_2026";
    std::string geometry_file;
    double geometry_x0 = 4.0;
    double geometry_y0 = 0.0;
    double geometry_scale = 1.0;
    double car_length = 5.50;      // [m], Re/Kolmogorov estimate 기준 길이
    double car_height = 1.15;      // [m], geometry bbox가 없을 때 Cd 기준 높이

    // 출력 디렉터리와 파일 prefix. VTK/CSV 이름을 case별로 분리한다.
    std::string output_dir = "output";
    std::string output_prefix = "f1";
};

// parser는 option과 함께 warning을 모아서 rank 0에서 사용자가 확인하게 한다.
struct ConfigResult {
    Options options;
    std::vector<std::string> warnings;
};

// INI 파일, CLI 인자, 기본 config writer를 담당하는 작은 유틸리티 클래스.
class ConfigParser {
public:
    static ConfigResult load(int argc, char** argv, int mpi_rank);
    static void write_default_config(const std::string& path);
    static void print_help(std::ostream& os, const char* exe);

private:
    static Options defaults();
    static void read_file(const std::string& path, std::unordered_map<std::string, std::string>& kv);
    static void apply_kv(Options& opt, const std::unordered_map<std::string, std::string>& kv,
                         std::vector<std::string>& warnings);
    static void apply_one(Options& opt, const std::string& key, const std::string& value,
                          std::vector<std::string>& warnings);
};

std::string trim_copy(const std::string& s);
bool parse_bool(const std::string& value);
