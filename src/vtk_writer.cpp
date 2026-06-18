#include "vtk_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

// Write ParaView-compatible legacy ASCII VTK directly without extra VTK libraries.
// Solver passes field names and pointers; format details are encapsulated here.
void write_structured_points_vtk(const std::string& path,
                                 int nx, int ny,
                                 double dx, double dy,
                                 const std::vector<VtkFieldView>& scalars,
                                 const std::vector<VtkVectorView>& vectors) {
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write VTK file: " + path);
    // STRUCTURED_POINTS is the simplest legacy VTK dataset for uniform Cartesian grids.
    out << "# vtk DataFile Version 3.0\n";
    out << "F1 side-view LES-LBM CFD\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << nx << ' ' << ny << " 1\n";
    out << "ORIGIN 0 0 0\n";
    out << std::setprecision(12) << "SPACING " << dx << ' ' << dy << " 1\n";
    out << "POINT_DATA " << n << "\n";

    out << std::setprecision(8);
    // Write scalar field; error out if data size is not nx*ny.
    for (const auto& field : scalars) {
        if (!field.data || field.data->size() != n) {
            throw std::runtime_error("Invalid scalar field size for VTK field: " + field.name);
        }
        out << "SCALARS " << field.name << " double 1\n";
        out << "LOOKUP_TABLE default\n";
        for (double v : *field.data) out << v << '\n';
    }
    // Write vector field; in 2D set z component to 0 for ParaView format.
    for (const auto& vec : vectors) {
        if (!vec.x || !vec.y || vec.x->size() != n || vec.y->size() != n) {
            throw std::runtime_error("Invalid vector field size for VTK vector: " + vec.name);
        }
        out << "VECTORS " << vec.name << " double\n";
        for (std::size_t i = 0; i < n; ++i) out << (*(vec.x))[i] << ' ' << (*(vec.y))[i] << " 0\n";
    }
}
