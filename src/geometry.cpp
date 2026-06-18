#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Vehicle geometry is modeled as a set of simple 2D primitives (polygon/circle/rectangle)
// instead of importing complex CAD/B-rep. This enables lightweight and deterministic
// solid masking for educational CFD studies.

namespace {

// Convert local (normalized) geometry coordinates into global simulator coordinates.
// The transform applies translation and global scale only once so all presets share
// a common shape definition path.
Point2 transform_point(double x, double y, const Options& opt) {
    return {opt.geometry_x0 + opt.geometry_scale * x,
            opt.geometry_y0 + opt.geometry_scale * y};
}

// Transform one dimension-only scalar in the same way as x/y transform.
double transform_len(double a, const Options& opt) { return opt.geometry_scale * a; }

// Classic ray-casting test for point-in-polygon membership.
// O(n) in vertex count; sufficient for the moderate-size polygons used here.
bool point_in_polygon(const std::vector<Point2>& poly, double x, double y) {
    bool inside = false;
    const std::size_t n = poly.size();
    if (n < 3) return false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const auto& pi = poly[i];
        const auto& pj = poly[j];

        // Check whether ray at y intersects segment (pi,pj).
        const bool crosses = ((pi.y > y) != (pj.y > y)) &&
            (x < (pj.x - pi.x) * (y - pi.y) / ((pj.y - pi.y) + 1.0e-300) + pi.x);
        if (crosses) inside = !inside;
    }
    return inside;
}

// Transform list of hard-coded local points for a preset using a small helper.
std::vector<Point2> transformed(std::initializer_list<std::array<double, 2>> pts, const Options& opt) {
    std::vector<Point2> out;
    out.reserve(pts.size());
    for (const auto& p : pts) out.push_back(transform_point(p[0], p[1], opt));
    return out;
}

// Parse all numeric tokens from an input stream.
// Geometry file parsing uses this helper to support polygon/circle/rect syntax.
std::vector<double> parse_numbers(std::istringstream& iss) {
    std::vector<double> values;
    double v = 0.0;
    while (iss >> v) values.push_back(v);
    return values;
}

} // anonymous namespace

// Expand bounding box with all points for one newly added primitive.
// This bbox is then used for drag normalization and summary output.
void Geometry2D::update_bbox(const Shape2D& s) {
    auto upd = [&](double x, double y) {
        min_x_ = std::min(min_x_, x);
        max_x_ = std::max(max_x_, x);
        min_y_ = std::min(min_y_, y);
        max_y_ = std::max(max_y_, y);
    };

    if (s.type == ShapeType::Circle) {
        // Circle extent is center ± radius.
        upd(s.cx - s.r, s.cy - s.r);
        upd(s.cx + s.r, s.cy + s.r);
    } else if (s.type == ShapeType::Rectangle) {
        // Rectangle stores two opposite corners.
        upd(s.points[0].x, s.points[0].y);
        upd(s.points[1].x, s.points[1].y);
    } else {
        // Polygon extent is over all vertices.
        for (const auto& p : s.points) upd(p.x, p.y);
    }
}

// Register polygon primitive after checking minimum vertex count.
// Any primitive with fewer than 3 points is invalid geometry input.
void Geometry2D::add_polygon(std::string name, std::vector<Point2> pts) {
    if (pts.size() < 3) throw std::runtime_error("Polygon needs at least 3 points: " + name);
    Shape2D s;
    s.type = ShapeType::Polygon;
    s.name = std::move(name);
    s.points = std::move(pts);
    update_bbox(s);
    shapes_.push_back(std::move(s));
}

// Register wheel/body circle primitive.
void Geometry2D::add_circle(std::string name, double cx, double cy, double r) {
    if (r <= 0.0) throw std::runtime_error("Circle radius must be positive: " + name);
    Shape2D s;
    s.type = ShapeType::Circle;
    s.name = std::move(name);
    s.cx = cx;
    s.cy = cy;
    s.r = r;
    update_bbox(s);
    shapes_.push_back(std::move(s));
}

// Register rectangle primitive and normalize opposite corners.
void Geometry2D::add_rect(std::string name, double xmin, double ymin, double xmax, double ymax) {
    // Ensure xmin<=xmax and ymin<=ymax so downstream checks can use direct comparisons.
    if (xmax < xmin) std::swap(xmin, xmax);
    if (ymax < ymin) std::swap(ymin, ymax);
    Shape2D s;
    s.type = ShapeType::Rectangle;
    s.name = std::move(name);
    s.points = {{xmin, ymin}, {xmax, ymax}};
    update_bbox(s);
    shapes_.push_back(std::move(s));
}

// Built-in side-view F1-like presets for comparison studies.
// Generic, low-drag, and high-downforce presets share the same topology with tuned local parameters.
Geometry2D Geometry2D::builtin(const Options& opt) {
    Geometry2D geom;
    geom.name_ = opt.geometry_preset.empty() ? "generic_f1_2026" : opt.geometry_preset;

    const bool low_drag = geom.name_ == "low_drag";
    const bool high_downforce = geom.name_ == "high_downforce";
    if (!(geom.name_ == "generic_f1_2026" || low_drag || high_downforce)) {
        throw std::runtime_error("Unknown geometry preset: " + geom.name_);
    }

    // Local side-view coordinate system:
    // x=0 at nose/leading edge, y=0 at ground reference.
    // Values are hand-tuned from proportions and non-proprietary shape heuristics.
    const double wing_front_h = high_downforce ? 0.17 : (low_drag ? 0.10 : 0.13);
    const double rear_wing_h0 = high_downforce ? 0.82 : (low_drag ? 0.88 : 0.84);
    const double rear_wing_h1 = high_downforce ? 1.18 : (low_drag ? 1.08 : 1.13);
    const double rear_wing_len = high_downforce ? 0.70 : (low_drag ? 0.48 : 0.60);
    const double body_peak = high_downforce ? 1.08 : (low_drag ? 0.97 : 1.02);
    const double diffuser = high_downforce ? 0.46 : (low_drag ? 0.30 : 0.38);

    // Main body / sidepod / engine cover / nose profile.
    geom.add_polygon("main_body", transformed({
        {0.18, 0.28}, {0.55, 0.36}, {0.95, 0.48}, {1.38, 0.55},
        {1.82, 0.69}, {2.12, 0.91}, {2.48, body_peak}, {2.95, 1.03},
        {3.35, 0.88}, {3.82, 0.69}, {4.42, 0.55}, {4.93, 0.47},
        {5.24, 0.55}, {5.46, 0.72}, {5.52, 0.83}, {5.28, 0.84},
        {4.88, 0.72}, {4.38, 0.52}, {3.62, 0.36}, {2.70, 0.27},
        {1.64, 0.27}, {0.78, 0.25}, {0.18, 0.28}
    }, opt));

    // Floor and diffuser for wake interaction near the bottom boundary.
    geom.add_polygon("floor_diffuser", transformed({
        {0.35, 0.12}, {3.70, 0.12}, {4.52, 0.16}, {5.30, diffuser},
        {5.36, diffuser + 0.10}, {4.70, 0.28}, {3.55, 0.21}, {0.35, 0.20}
    }, opt));

    // Nose bridge and thin front wing representation.
    geom.add_polygon("nose", transformed({
        {0.38, 0.33}, {1.22, 0.42}, {1.95, 0.58}, {2.05, 0.66},
        {1.22, 0.54}, {0.35, 0.42}
    }, opt));

    // Front wing approximated as two thin polygons.
    geom.add_polygon("front_wing_main", transformed({
        {-0.08, 0.05}, {1.08, 0.05}, {1.17, wing_front_h}, {0.05, wing_front_h + 0.02}
    }, opt));
    geom.add_polygon("front_wing_flap", transformed({
        {0.04, wing_front_h + 0.05}, {0.98, wing_front_h + 0.07},
        {1.04, wing_front_h + 0.13}, {0.00, wing_front_h + 0.11}
    }, opt));

    // Rear wing and pylon. Rear wing size changes with preset objective.
    geom.add_polygon("rear_wing_main", transformed({
        {5.02, rear_wing_h0}, {5.02 + rear_wing_len, rear_wing_h0 + 0.02},
        {5.00 + rear_wing_len, rear_wing_h0 + 0.13}, {4.96, rear_wing_h0 + 0.11}
    }, opt));
    geom.add_polygon("rear_wing_upper", transformed({
        {4.96, rear_wing_h1}, {5.00 + rear_wing_len, rear_wing_h1 + 0.01},
        {4.98 + rear_wing_len, rear_wing_h1 + 0.11}, {4.92, rear_wing_h1 + 0.10}
    }, opt));
    geom.add_rect("rear_wing_pylon", transform_point(5.12, 0.52, opt).x,
                 transform_point(5.12, 0.52, opt).y,
                 transform_point(5.20, rear_wing_h1, opt).x,
                 transform_point(5.20, rear_wing_h1, opt).y);

    // Wheels represented as circles in 2D side view.
    const double r_front_local = low_drag ? 0.335 : 0.355;
    const double r_rear_local = high_downforce ? 0.375 : 0.365;
    const double r_front = transform_len(r_front_local, opt);
    const double r_rear = transform_len(r_rear_local, opt);
    const auto fc = transform_point(1.16, r_front_local + 0.005, opt);
    const auto rc = transform_point(4.56, r_rear_local + 0.005, opt);
    geom.add_circle("front_wheel", fc.x, fc.y, r_front);
    geom.add_circle("rear_wheel", rc.x, rc.y, r_rear);

    // Side-view suspension struts and supports.
    geom.add_polygon("front_suspension_upper", transformed({
        {1.08, 0.66}, {1.18, 0.66}, {1.62, 0.48}, {1.58, 0.42}
    }, opt));
    geom.add_polygon("front_suspension_lower", transformed({
        {1.02, 0.24}, {1.12, 0.24}, {1.60, 0.36}, {1.55, 0.42}
    }, opt));
    geom.add_polygon("rear_suspension_upper", transformed({
        {4.46, 0.64}, {4.56, 0.64}, {4.05, 0.48}, {4.08, 0.42}
    }, opt));
    geom.add_polygon("rear_suspension_lower", transformed({
        {4.52, 0.24}, {4.62, 0.24}, {4.10, 0.36}, {4.06, 0.42}
    }, opt));

    if (geom.shapes_.empty()) throw std::runtime_error("Generated empty geometry");
    return geom;
}

// Parse external .geom description.
// Input syntax:
//   polygon x1 y1 x2 y2 ...
//   circle cx cy r
//   rect xmin ymin xmax ymax
//   name string comment
Geometry2D Geometry2D::from_file(const Options& opt, const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open geometry file: " + path);

    Geometry2D geom;
    geom.name_ = path;
    std::string line;
    int line_no = 0;

    while (std::getline(in, line)) {
        ++line_no;

        // Remove inline comments for robust manual edits.
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim_copy(line);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string type;
        iss >> type;
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return std::tolower(c); });

        // Auto-generate a stable primitive name from the source line number.
        std::string name = type + "_" + std::to_string(line_no);

        if (type == "name") {
            std::string rest;
            std::getline(iss, rest);
            geom.name_ = trim_copy(rest);
            continue;
        }

        const auto vals = parse_numbers(iss);

        try {
            if (type == "polygon") {
                // At least 3 points, even number of scalars.
                if (vals.size() < 6 || vals.size() % 2 != 0) {
                    throw std::runtime_error("polygon requires even x y pairs, at least 3 points");
                }
                std::vector<Point2> pts;
                for (std::size_t i = 0; i < vals.size(); i += 2)
                    pts.push_back(transform_point(vals[i], vals[i + 1], opt));
                geom.add_polygon(name, std::move(pts));
            } else if (type == "circle") {
                if (vals.size() != 3) throw std::runtime_error("circle requires cx cy r");
                const auto c = transform_point(vals[0], vals[1], opt);
                geom.add_circle(name, c.x, c.y, transform_len(vals[2], opt));
            } else if (type == "rect" || type == "rectangle") {
                if (vals.size() != 4) throw std::runtime_error("rect requires xmin ymin xmax ymax");
                const auto a = transform_point(vals[0], vals[1], opt);
                const auto b = transform_point(vals[2], vals[3], opt);
                geom.add_rect(name, a.x, a.y, b.x, b.y);
            } else {
                throw std::runtime_error("unknown primitive: " + type);
            }
        } catch (const std::exception& e) {
            std::ostringstream err;
            err << path << ':' << line_no << ": " << e.what();
            throw std::runtime_error(err.str());
        }
    }

    if (geom.shapes_.empty()) throw std::runtime_error("Geometry file contains no primitives: " + path);
    return geom;
}

// A cell center is solid if any primitive contains it.
// The union operation is short-circuited: as soon as one primitive matches, true is returned.
bool Geometry2D::contains(double x, double y) const {
    for (const auto& s : shapes_) {
        if (s.type == ShapeType::Circle) {
            // Circle test uses distance squared to avoid sqrt.
            const double dx = x - s.cx;
            const double dy = y - s.cy;
            if (dx * dx + dy * dy <= s.r * s.r) return true;
        } else if (s.type == ShapeType::Rectangle) {
            // Axis-aligned rectangle check using prepared corners.
            if (x >= s.points[0].x && x <= s.points[1].x && y >= s.points[0].y && y <= s.points[1].y) return true;
        } else {
            // Polygon containment handles irregular side-body and aero-body details.
            if (point_in_polygon(s.points, x, y)) return true;
        }
    }
    return false;
}

// Export only geometric outlines as lines.
// VTK output is for visual validation; it does not encode solver field values.
void Geometry2D::write_legacy_vtk(const std::string& path) const {
    std::vector<Point2> points;
    std::vector<std::array<int, 2>> lines;

    // Append points for one closed contour and create connectivity for line cells.
    auto add_line_loop = [&](const std::vector<Point2>& pts) {
        const int base = static_cast<int>(points.size());
        points.insert(points.end(), pts.begin(), pts.end());
        const int n = static_cast<int>(pts.size());
        for (int i = 0; i < n; ++i) lines.push_back({base + i, base + ((i + 1) % n)});
    };

    for (const auto& s : shapes_) {
        if (s.type == ShapeType::Circle) {
            // Circles are approximated by a fixed 96-segment polygon in VTK only.
            std::vector<Point2> pts;
            constexpr int seg = 96;
            constexpr double pi = 3.141592653589793238462643383279502884;
            for (int k = 0; k < seg; ++k) {
                const double a = 2.0 * pi * static_cast<double>(k) / static_cast<double>(seg);
                pts.push_back({s.cx + s.r * std::cos(a), s.cy + s.r * std::sin(a)});
            }
            add_line_loop(pts);
        } else if (s.type == ShapeType::Rectangle) {
            // Use explicit 4 corners to avoid angle ordering ambiguity.
            const auto a = s.points[0];
            const auto b = s.points[1];
            add_line_loop({{a.x, a.y}, {b.x, a.y}, {b.x, b.y}, {a.x, b.y}});
        } else {
            add_line_loop(s.points);
        }
    }

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write geometry VTK: " + path);

    out << "# vtk DataFile Version 3.0\n";
    out << "F1 side-view geometry outline: " << name_ << "\n";
    out << "ASCII\nDATASET POLYDATA\n";
    out << "POINTS " << points.size() << " double\n";
    out << std::setprecision(12);
    for (const auto& p : points) out << p.x << ' ' << p.y << " 0\n";
    out << "LINES " << lines.size() << ' ' << lines.size() * 3 << "\n";
    for (const auto& l : lines) out << "2 " << l[0] << ' ' << l[1] << "\n";
}
