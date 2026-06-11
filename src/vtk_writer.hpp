#pragma once

#include <string>
#include <vector>

// ParaView legacy VTK writer가 참조할 scalar field view.
// data는 nx*ny 크기의 전역 field를 가리켜야 한다.
struct VtkFieldView {
    std::string name;
    const std::vector<double>* data = nullptr;
};

// 2차원 벡터 field view. VTK에는 z=0을 붙여 3성분 벡터로 저장한다.
struct VtkVectorView {
    std::string name;
    const std::vector<double>* x = nullptr;
    const std::vector<double>* y = nullptr;
};

// ASCII legacy STRUCTURED_POINTS VTK 파일을 쓴다.
// 별도 라이브러리 없이 ParaView 호환 출력을 만들기 위한 최소 writer이다.
void write_structured_points_vtk(const std::string& path,
                                 int nx, int ny,
                                 double dx, double dy,
                                 const std::vector<VtkFieldView>& scalars,
                                 const std::vector<VtkVectorView>& vectors);
