#pragma once

#include "config.hpp"

#include <array>
#include <iosfwd>
#include <string>
#include <vector>

// 2차원 물리 좌표점. 모든 geometry primitive는 meter 단위 좌표를 사용한다.
struct Point2 {
    double x = 0.0;
    double y = 0.0;
};

// 차량 형상을 구성하는 기본 primitive 종류.
enum class ShapeType { Polygon, Circle, Rectangle };

// 하나의 solid primitive. polygon/rectangle은 points를 쓰고, circle은 cx/cy/r을 쓴다.
struct Shape2D {
    ShapeType type = ShapeType::Polygon;
    std::string name;
    std::vector<Point2> points; // polygon 꼭짓점 또는 rectangle의 [min,max] corner
    double cx = 0.0;
    double cy = 0.0;
    double r = 0.0;
};

// Geometry2D는 여러 primitive의 union으로 차량 solid 영역을 표현한다.
// solver는 각 cell center가 contains(x,y)에 포함되는지만 확인하면 되므로,
// 실제 CAD가 없어도 간단한 형상 교체 실험이 가능하다.
class Geometry2D {
public:
    // 코드 내부 F1 preset 생성.
    static Geometry2D builtin(const Options& opt);
    // examples/*.geom 같은 외부 primitive 파일 읽기.
    static Geometry2D from_file(const Options& opt, const std::string& path);

    // 주어진 물리 좌표가 solid 차량 내부인지 판정한다.
    bool contains(double x, double y) const;
    // ParaView에서 형상만 따로 확인할 수 있는 geometry VTK 출력.
    void write_legacy_vtk(const std::string& path) const;

    const std::vector<Shape2D>& shapes() const { return shapes_; }
    const std::string& name() const { return name_; }
    double min_x() const { return min_x_; }
    double max_x() const { return max_x_; }
    double min_y() const { return min_y_; }
    double max_y() const { return max_y_; }
    double length() const { return max_x_ > min_x_ ? max_x_ - min_x_ : 0.0; }
    double height() const { return max_y_ > min_y_ ? max_y_ - min_y_ : 0.0; }

private:
    std::string name_ = "geometry";
    std::vector<Shape2D> shapes_;
    // 항력 기준 높이 및 디버그 출력을 위한 bounding box.
    double min_x_ = 1.0e300;
    double max_x_ = -1.0e300;
    double min_y_ = 1.0e300;
    double max_y_ = -1.0e300;

    void add_polygon(std::string name, std::vector<Point2> pts);
    void add_circle(std::string name, double cx, double cy, double r);
    void add_rect(std::string name, double xmin, double ymin, double xmax, double ymax);
    void update_bbox(const Shape2D& s);
};
