#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Central configuration pipeline:
// - start from hard-coded defaults (deterministic baseline),
// - apply INI file entries in section.key form,
// - then apply CLI "--set" overrides in call order,
// - finally adjust derived settings (auto mesh) and run validity checks.
// This guarantees that bad settings are rejected before heavy domain decomposition or MPI startup.

namespace {

// Parse "double" values from INI/CLI text and validate full token consumption.
// We intentionally reject partial parses so accidental suffixes like "10m" are caught
// instead of silently accepted as 10 and producing confusing physical units.
double to_double(const std::string& v, const std::string& key) {
    char* end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0' || !std::isfinite(x)) {
        throw std::runtime_error("Invalid floating value for " + key + ": " + v);
    }
    return x;
}

int to_int(const std::string& v, const std::string& key) {
    char* end = nullptr;
    const long x = std::strtol(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0') {
        throw std::runtime_error("Invalid integer value for " + key + ": " + v);
    }
    return static_cast<int>(x);
}

std::uint64_t to_u64(const std::string& v, const std::string& key) {
    char* end = nullptr;
    const unsigned long long x = std::strtoull(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0') {
        throw std::runtime_error("Invalid unsigned integer value for " + key + ": " + v);
    }
    return static_cast<std::uint64_t>(x);
}

// Normalize section/key/value string tokens to lower case for case-insensitive matching.
// Geometry names and solver names are treated as case-insensitive by design.
std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Optionally regenerate mesh from the LES target Kolmogorov scale.
// LES guideline uses η ≈ L * Re^(-3/4) (here L is car_length), and sets
// cell size h ≈ les_delta_to_eta * η.  This creates a physically motivated mesh.
// The current final cases keep auto_mesh_from_kolmogorov=false, but this path remains
// available for quick resolution scaling studies and reproducibility checks.
void set_auto_mesh(Options& opt, std::vector<std::string>& warnings) {
    if (!opt.auto_mesh_from_kolmogorov) {
        return;
    }
    const double re = std::max(1.0, opt.reynolds);
    const double eta = opt.car_length * std::pow(re, -0.75);
    const double target_h = std::max(eta * opt.les_delta_to_eta, 1.0e-9);
    const int nx = static_cast<int>(std::ceil(opt.domain_length / target_h)) + 1;
    const int ny = static_cast<int>(std::ceil(opt.domain_height / target_h)) + 1;
    const std::uint64_t cells = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny);
    std::ostringstream msg;
    msg << std::setprecision(6)
        << "Kolmogorov estimate eta=" << eta << " m, LES target dx≈" << target_h
        << " m gives mesh " << nx << "x" << ny << " (" << cells << " cells).";
    warnings.push_back(msg.str());
    if (cells > opt.max_cells && !opt.allow_large_mesh) {
        std::ostringstream err;
        err << "auto mesh would allocate " << cells << " cells, exceeding mesh.max_cells="
            << opt.max_cells << ". Increase max_cells or set mesh.allow_large_mesh=true.";
        throw std::runtime_error(err.str());
    }
    opt.nx = nx;
    opt.ny = ny;
}

// Core safety checks that reject invalid configuration and issue warnings for known risks.
// The checks focus on solver contract (method options), runtime constraints
// (positive step counts), and LBM/FVM stability assumptions.
void sanity_check(Options& opt, std::vector<std::string>& warnings) {
    if (opt.solver_method != "fvm" && opt.solver_method != "lbm") {
        throw std::runtime_error("solver.method must be either fvm or lbm");
    }
    if (opt.nx < 16 || opt.ny < 16) {
        throw std::runtime_error("mesh.nx and mesh.ny must both be >= 16");
    }
    if (opt.steps < 0 || opt.output_interval <= 0 || opt.drag_interval <= 0 || opt.log_interval <= 0) {
        throw std::runtime_error("time/interval options must be positive (steps may be 0)");
    }
    if (opt.domain_length <= 0.0 || opt.domain_height <= 0.0) {
        throw std::runtime_error("domain.length and domain.height must be positive");
    }
    if (opt.inlet_velocity <= 0.0 || opt.density <= 0.0) {
        throw std::runtime_error("flow.u_in and flow.rho must be positive");
    }
    if (opt.dynamic_viscosity <= 0.0 && opt.reynolds <= 0.0) {
        throw std::runtime_error("Set either flow.mu > 0 or flow.reynolds > 0");
    }
    if (opt.cfl <= 0.0) {
        throw std::runtime_error("flow.cfl must be positive");
    }
    if (opt.solver_method == "lbm" && opt.cfl > 0.2) {
        warnings.push_back("LBM is weakly compressible; cfl/lattice Mach above ~0.1 can be inaccurate or unstable.");
    }
    if (opt.solver_method == "fvm" && opt.cfl > 0.5) {
        warnings.push_back("Explicit FVM mode is usually run with flow.cfl <= 0.5.");
    }
    const double dx = opt.domain_length / static_cast<double>(opt.nx - 1);
    const double dy = opt.domain_height / static_cast<double>(opt.ny - 1);
    const double rel = std::abs(dx - dy) / std::max(dx, dy);
    if (rel > 0.05) {
        std::ostringstream msg;
        msg << "LBM assumes nearly square cells, but dx=" << dx << " m and dy=" << dy
            << " m. Adjust nx/ny for higher accuracy.";
        warnings.push_back(msg.str());
    }
    const std::uint64_t cells = static_cast<std::uint64_t>(opt.nx) * static_cast<std::uint64_t>(opt.ny);
    if (cells > opt.max_cells && !opt.allow_large_mesh) {
        std::ostringstream err;
        err << "mesh has " << cells << " cells, exceeding mesh.max_cells=" << opt.max_cells
            << ". Increase max_cells or set mesh.allow_large_mesh=true.";
        throw std::runtime_error(err.str());
    }
    if (opt.fvm_pressure_iterations <= 0 || opt.fvm_pressure_omega <= 0.0 || opt.fvm_pressure_omega > 1.0) {
        throw std::runtime_error("FVM pressure iterations must be positive and pressure omega must be in (0,1]");
    }
}

} // anonymous namespace

// Trim helper used by both INI file and CLI parsing.
// Trims both leading and trailing ASCII whitespace, including tabs and spaces.
std::string trim_copy(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// Parse booleans with a compact user-friendly set of aliases.
// This reduces typing friction for config tweaks while still validating invalid tokens.
bool parse_bool(const std::string& value) {
    const std::string v = lower_copy(trim_copy(value));
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw std::runtime_error("Invalid boolean value: " + value);
}

// Return a copy of compile-time default configuration.
// Keeping this as a function makes adding new default fields safer than global state.
Options ConfigParser::defaults() { return Options{}; }

// Print help text to a chosen output stream.
// In MPI execution only rank 0 should call this to avoid interleaved output.
void ConfigParser::print_help(std::ostream& os, const char* exe) {
    os << "Usage: " << exe << " [--config path] [--set section.key=value] [--write-default-config path]\n"
       << "       " << exe << " --list-geometry\n\n"
       << "Main sections: case, domain, mesh, flow, les, time, geometry, output.\n"
       << "Example: mpirun -n 4 ./f1_cfd --config examples/f1_demo.ini --set time.steps=5000\n";
}

// Write a default INI template to a path chosen by the caller.
// This acts as an official documented baseline for reproducible reruns.
void ConfigParser::write_default_config(const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write default config to " + path);
    out << R"CFG(# F1 side-view LES-LBM CFD configuration
[case]
name = f1_demo

[solver]
method = lbm        # verified high-Re path; fvm is available as experimental

[domain]
length = 18.0       # m, streamwise
height = 4.5        # m, vertical side-view domain

[mesh]
nx = 1081
ny = 271
auto_from_kolmogorov = false
les_delta_to_eta = 20.0
max_cells = 20000000
allow_large_mesh = false

[flow]
u_in = 30.0         # m/s
rho = 1.225         # kg/m^3
mu = 0.0            # Pa s; if <=0 use Reynolds number
reynolds = 2000.0   # based on geometry.car_length; increase with matching mesh refinement
cfl = 0.15          # fixed; solver never changes dt after startup
perturbation = 0.001

[les]
smagorinsky_cs = 0.30
filter_width_cells = 1.0
tau_min = 0.5001
tau_max = 2.5

[fvm]
pressure_iterations = 120
pressure_tolerance = 1e-7
pressure_omega = 0.85
nu_eff_max_factor = 200.0

[time]
steps = 240000          # 20 s for default fixed dt=8.333333e-5 s
output_interval = 1200  # write VTK every 0.1 s for default fixed dt
drag_interval = 1200
log_interval = 1200

[geometry]
preset = generic_f1_2026  # generic_f1_2026, low_drag, high_downforce
file =                  # optional custom geometry file overrides preset
x0 = 4.0
y0 = 0.0
scale = 1.0
car_length = 5.50
car_height = 1.15

[output]
directory = output
prefix = f1
)CFG";
}

// Parse a simple INI syntax:
// - lines "key = value" become entries,
// - [section] changes the implicit prefix,
// - comments after '#' are stripped.
// Flat map keys are stored as "section.key" to keep downstream parser logic simple.
void ConfigParser::read_file(const std::string& path, std::unordered_map<std::string, std::string>& kv) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open config file: " + path);
    std::string section;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim_copy(line);
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            section = lower_copy(trim_copy(line.substr(1, line.size() - 2)));
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            std::ostringstream err;
            err << path << ':' << line_no << ": expected key=value";
            throw std::runtime_error(err.str());
        }
        std::string key = lower_copy(trim_copy(line.substr(0, eq)));
        std::string value = trim_copy(line.substr(eq + 1));
        if (!section.empty() && key.find('.') == std::string::npos) key = section + "." + key;
        kv[key] = value;
    }
}

// Apply one parsed key-value pair into Options.
// Unknown keys are intentionally downgraded into runtime warnings so experiments can
// keep legacy config files that contain extra/obsolete entries without hard failure.
void ConfigParser::apply_one(Options& opt, const std::string& key_raw, const std::string& value,
                             std::vector<std::string>& warnings) {
    const std::string key = lower_copy(trim_copy(key_raw));
    if (key == "case.name") opt.case_name = value;
    else if (key == "solver.method") {
        opt.solver_method = lower_copy(value);
    }
    else if (key == "domain.length") opt.domain_length = to_double(value, key);
    else if (key == "domain.height") opt.domain_height = to_double(value, key);
    else if (key == "mesh.nx") opt.nx = to_int(value, key);
    else if (key == "mesh.ny") opt.ny = to_int(value, key);
    else if (key == "mesh.auto_from_kolmogorov") opt.auto_mesh_from_kolmogorov = parse_bool(value);
    else if (key == "mesh.les_delta_to_eta") opt.les_delta_to_eta = to_double(value, key);
    else if (key == "mesh.max_cells") opt.max_cells = to_u64(value, key);
    else if (key == "mesh.allow_large_mesh") opt.allow_large_mesh = parse_bool(value);
    else if (key == "flow.u_in") opt.inlet_velocity = to_double(value, key);
    else if (key == "flow.rho") opt.density = to_double(value, key);
    else if (key == "flow.mu") opt.dynamic_viscosity = to_double(value, key);
    else if (key == "flow.reynolds") opt.reynolds = to_double(value, key);
    else if (key == "flow.cfl") opt.cfl = to_double(value, key);
    else if (key == "flow.perturbation") opt.perturbation = to_double(value, key);
    else if (key == "les.smagorinsky_cs") opt.smagorinsky_cs = to_double(value, key);
    else if (key == "les.filter_width_cells") opt.les_filter_width_cells = to_double(value, key);
    else if (key == "les.tau_min") opt.tau_min = to_double(value, key);
    else if (key == "les.tau_max") opt.tau_max = to_double(value, key);
    else if (key == "fvm.pressure_iterations") opt.fvm_pressure_iterations = to_int(value, key);
    else if (key == "fvm.pressure_tolerance") opt.fvm_pressure_tolerance = to_double(value, key);
    else if (key == "fvm.pressure_omega") opt.fvm_pressure_omega = to_double(value, key);
    else if (key == "fvm.nu_eff_max_factor") opt.fvm_nu_eff_max_factor = to_double(value, key);
    else if (key == "time.steps") opt.steps = to_int(value, key);
    else if (key == "time.output_interval") opt.output_interval = to_int(value, key);
    else if (key == "time.drag_interval") opt.drag_interval = to_int(value, key);
    else if (key == "time.log_interval") opt.log_interval = to_int(value, key);
    else if (key == "geometry.preset") opt.geometry_preset = lower_copy(value);
    else if (key == "geometry.file") opt.geometry_file = value;
    else if (key == "geometry.x0") opt.geometry_x0 = to_double(value, key);
    else if (key == "geometry.y0") opt.geometry_y0 = to_double(value, key);
    else if (key == "geometry.scale") opt.geometry_scale = to_double(value, key);
    else if (key == "geometry.car_length") opt.car_length = to_double(value, key);
    else if (key == "geometry.car_height") opt.car_height = to_double(value, key);
    else if (key == "output.directory") opt.output_dir = value;
    else if (key == "output.prefix") opt.output_prefix = value;
    else warnings.push_back("Ignoring unknown config key: " + key);
}

// Copy every config entry from an unordered map into Options.
// Iteration order is map-iteration order here; deterministic order is intentionally
// not required because all values are independent assignments.
void ConfigParser::apply_kv(Options& opt, const std::unordered_map<std::string, std::string>& kv,
                            std::vector<std::string>& warnings) {
    for (const auto& [key, value] : kv) apply_one(opt, key, value, warnings);
}

// Full configuration loading pipeline:
// 1) Start from defaults to guarantee every option has a valid baseline.
// 2) Load INI entries when --config is given.
// 3) Collect --set overrides in CLI order and apply after file load.
// 4) Execute derivations/checks (mesh estimation, sanity checks) before returning.
//   A fully prepared ConfigResult is returned only if no fatal issues were found.
ConfigResult ConfigParser::load(int argc, char** argv, int mpi_rank) {
    Options opt = defaults();
    std::vector<std::string> warnings;
    std::unordered_map<std::string, std::string> file_kv;
    std::vector<std::pair<std::string, std::string>> overrides;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        // Print usage and exit early; no solver run should start after --help.
        if (arg == "--help" || arg == "-h") {
            if (mpi_rank == 0) print_help(std::cout, argv[0]);
            std::exit(0);
        // Load one optional config file; repeated use can merge later files if needed.
        } else if (arg == "--config" || arg == "-c") {
            if (++i >= argc) throw std::runtime_error("--config requires a path");
            read_file(argv[i], file_kv);
        // Runtime override syntax: --set key=value.
        // These always win over file values, and can be used to quickly sweep parameters.
        } else if (arg == "--set") {
            if (++i >= argc) throw std::runtime_error("--set requires key=value");
            const std::string kv = argv[i];
            const auto eq = kv.find('=');
            if (eq == std::string::npos) throw std::runtime_error("--set requires key=value");
            overrides.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        // Dump a reference configuration template and terminate.
        // Useful for CI bootstrap and reproducible test-case capture.
        } else if (arg == "--write-default-config") {
            if (++i >= argc) throw std::runtime_error("--write-default-config requires a path");
            if (mpi_rank == 0) write_default_config(argv[i]);
            std::exit(0);
        // Print supported built-in geometry presets and primitive-file syntax.
        } else if (arg == "--list-geometry") {
            if (mpi_rank == 0) {
                std::cout << "Built-in presets:\n"
                          << "  generic_f1_2026\n  low_drag\n  high_downforce\n"
                          << "Custom file primitives: polygon x1 y1 x2 y2 ..., circle cx cy r, rect xmin ymin xmax ymax\n";
            }
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    apply_kv(opt, file_kv, warnings);
    for (const auto& [k, v] : overrides) apply_one(opt, k, v, warnings);
    set_auto_mesh(opt, warnings);
    sanity_check(opt, warnings);
    return {opt, warnings};
}
