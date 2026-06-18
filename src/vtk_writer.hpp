#pragma once

#include <string>
#include <vector>

// Scalar field views referenced by the legacy ParaView VTK writer.
// Data must reference a global field with size nx*ny.
struct VtkFieldView {
    std::string name;
    const std::vector<double>* data = nullptr;
};

// 2D vector field view; store as 3-component vector with z=0.
struct VtkVectorView {
    std::string name;
    const std::vector<double>* x = nullptr;
    const std::vector<double>* y = nullptr;
};

// Write ASCII legacy STRUCTURED_POINTS VTK files.
// Minimal writer for ParaView-compatible output without extra libraries.
void write_structured_points_vtk(const std::string& path,
                                 int nx, int ny,
                                 double dx, double dy,
                                 const std::vector<VtkFieldView>& scalars,
                                 const std::vector<VtkVectorView>& vectors);
