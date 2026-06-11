#include "vtk_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

// 별도 VTK 라이브러리 없이 ParaView가 읽을 수 있는 legacy ASCII VTK를 직접 쓴다.
// solver 쪽에서는 field 이름과 포인터만 넘기고, 실제 파일 포맷 세부사항은 여기로 숨겼다.
void write_structured_points_vtk(const std::string& path,
                                 int nx, int ny,
                                 double dx, double dy,
                                 const std::vector<VtkFieldView>& scalars,
                                 const std::vector<VtkVectorView>& vectors) {
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write VTK file: " + path);
    // STRUCTURED_POINTS는 균일 직교격자에 적합한 가장 단순한 legacy VTK dataset이다.
    out << "# vtk DataFile Version 3.0\n";
    out << "F1 side-view LES-LBM CFD\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << nx << ' ' << ny << " 1\n";
    out << "ORIGIN 0 0 0\n";
    out << std::setprecision(12) << "SPACING " << dx << ' ' << dy << " 1\n";
    out << "POINT_DATA " << n << "\n";

    out << std::setprecision(8);
    // scalar field 출력. data 크기가 nx*ny와 다르면 즉시 오류를 내서 깨진 VTK를 방지한다.
    for (const auto& field : scalars) {
        if (!field.data || field.data->size() != n) {
            throw std::runtime_error("Invalid scalar field size for VTK field: " + field.name);
        }
        out << "SCALARS " << field.name << " double 1\n";
        out << "LOOKUP_TABLE default\n";
        for (double v : *field.data) out << v << '\n';
    }
    // vector field 출력. 2D 계산이지만 ParaView vector 형식에 맞춰 z 성분은 0으로 둔다.
    for (const auto& vec : vectors) {
        if (!vec.x || !vec.y || vec.x->size() != n || vec.y->size() != n) {
            throw std::runtime_error("Invalid vector field size for VTK vector: " + vec.name);
        }
        out << "VECTORS " << vec.name << " double\n";
        for (std::size_t i = 0; i < n; ++i) out << (*(vec.x))[i] << ' ' << (*(vec.y))[i] << " 0\n";
    }
}
