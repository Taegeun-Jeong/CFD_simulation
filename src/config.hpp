#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
// Central configuration object consumed by all solver paths.
// Values are assembled in this order:
//   1) compiled-in defaults from Options{} (base reproducible baseline),
//   2) entries loaded from INI file, and
//   3) --set key=value overrides.
// Every field is intentionally explicit so all physical/numerical knobs are discoverable
// from a single type declaration.
struct Options {
    // Human-readable label for logs, output file names, and README cross-check.
    std::string case_name = "f1_demo";

    // Solver switch: "lbm" (main production path) or "fvm" (research/extension path).
    std::string solver_method = "lbm";

    // ---------------------------------------------------------------------
    // Domain geometry in SI units (m).
    // ---------------------------------------------------------------------
    double domain_length = 18.0;   // Streamwise tunnel length
    double domain_height = 4.5;    // Vertical tunnel height

    // ---------------------------------------------------------------------
    // Mesh and LES resolution.
    // If auto_mesh_from_kolmogorov=true, nx/ny are recomputed from eta estimate.
    // Otherwise, these values are used as-is.
    // ---------------------------------------------------------------------
    int nx = 361;
    int ny = 91;
    bool auto_mesh_from_kolmogorov = false;
    double les_delta_to_eta = 20.0;
    std::uint64_t max_cells = 20000000ULL;
    bool allow_large_mesh = false;

    // ---------------------------------------------------------------------
    // Fluid properties and fixed-step control.
    // If dynamic_viscosity <= 0, viscosity is inferred from Reynolds input.
    // dt is computed once from CFL at startup and kept constant.
    // ---------------------------------------------------------------------
    double inlet_velocity = 30.0;  // [m/s], upstream freestream speed
    double density = 1.225;        // [kg/m^3], ambient density
    double dynamic_viscosity = 0.0;// [Pa s], positive -> explicit mu, non-positive -> use Reynolds
    double reynolds = 2000.0;      // Re based on geometry.car_length
    double cfl = 0.15;             // Fixed CFL target (dt is not updated in time)
    double perturbation = 0.001;   // Small velocity perturbation amplitude for startup development

    // ---------------------------------------------------------------------
    // LES and lattice relaxation controls.
    // Smagorinsky parameters are used by both LBM/FVM paths.
    // ---------------------------------------------------------------------
    double smagorinsky_cs = 0.30;
    double les_filter_width_cells = 1.0;
    double tau_min = 0.5001;
    double tau_max = 2.5;

    // ---------------------------------------------------------------------
    // FVM-only knobs (projection Poisson + viscosity capping).
    // ---------------------------------------------------------------------
    int fvm_pressure_iterations = 120;
    double fvm_pressure_tolerance = 1.0e-7;
    double fvm_pressure_omega = 0.85;
    double fvm_nu_eff_max_factor = 200.0;

    // ---------------------------------------------------------------------
    // Time loop and output cadence (step counters).
    // dt is derived before the loop; these are pure step counts.
    // ---------------------------------------------------------------------
    int steps = 240000;          // 20 s for dx=1/60 m, U=30 m/s, CFL=0.15
    int output_interval = 1200;  // default VTK interval = 0.1 s with fixed dt
    int drag_interval = 1200;
    int log_interval = 1200;

    // ---------------------------------------------------------------------
    // Geometry placement and reference scales.
    // Preset names are loaded from Geometry2D::builtin(); custom file geometry via geometry_file.
    // ---------------------------------------------------------------------
    std::string geometry_preset = "generic_f1_2026";
    std::string geometry_file;
    double geometry_x0 = 4.0;
    double geometry_y0 = 0.0;
    double geometry_scale = 1.0;
    double car_length = 5.50;      // [m], reference length for Re/eta estimate
    double car_height = 1.15;      // [m], fallback reference height if bbox missing

    // ---------------------------------------------------------------------
    // Output pathing for case separation.
    // Files: <output_dir>/<output_prefix>_*.vtk/.csv
    // ---------------------------------------------------------------------
    std::string output_dir = "output";
    std::string output_prefix = "f1";
};

// ---------------------------------------------------------------------------
// Parsed result with optional user-facing warnings.
// Warnings are shown by rank 0 but do not abort immediately.
// ---------------------------------------------------------------------------
struct ConfigResult {
    Options options;
    std::vector<std::string> warnings;
};

// ---------------------------------------------------------------------------
// ConfigParser utility: INI parser, CLI parser, and schema validation helpers.
// ---------------------------------------------------------------------------
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

// Trim whitespace utility shared by parser and geometry readers.
std::string trim_copy(const std::string& s);

// User-friendly boolean parser with alias values (true/false, yes/no, on/off, 1/0).
bool parse_bool(const std::string& value);
