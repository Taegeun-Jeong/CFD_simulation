#pragma once

#include "config.hpp"

#include <array>
#include <iosfwd>
#include <string>
#include <vector>

// 2D Euclidean point used by all geometry primitives.
// All coordinates passed here are in simulator meters.
struct Point2 {
    double x = 0.0;
    double y = 0.0;
};

// Primitive shape kinds that can compose a 2D body mask.
enum class ShapeType { Polygon, Circle, Rectangle };

// Shape2D is the immutable atomic geometry object stored in a Geometry2D union.
// - Polygon: arbitrary closed chain represented by vertices
// - Circle: center + radius
// - Rectangle: two corner points
struct Shape2D {
    ShapeType type = ShapeType::Polygon;
    std::string name;                  // Human-readable primitive label
    std::vector<Point2> points;        // vertices (polygon) or [min,max] corners (rectangle)
    double cx = 0.0;                  // circle center x
    double cy = 0.0;                  // circle center y
    double r = 0.0;                   // circle radius
};

// Geometry2D stores the entire vehicle region as a union of Shape2D objects.
// Solvers call contains(x,y) for each sampled grid point to create a solid mask.
// The class is file-independent and can be replaced by adding new geometry primitives.
class Geometry2D {
public:
    // Construct one of the built-in side-view F1-like presets.
    static Geometry2D builtin(const Options& opt);
    // Read geometry description from plain text primitive file (examples/*.geom).
    static Geometry2D from_file(const Options& opt, const std::string& path);

    // Test whether point (x, y) lies inside any primitive.
    bool contains(double x, double y) const;
    // Export outline-only geometry for standalone ParaView inspection.
    void write_legacy_vtk(const std::string& path) const;

    // Immutable external accessors used for summaries and drag scaling.
    const std::vector<Shape2D>& shapes() const { return shapes_; }
    const std::string& name() const { return name_; }
    double min_x() const { return min_x_; }
    double max_x() const { return max_x_; }
    double min_y() const { return min_y_; }
    double max_y() const { return max_y_; }
    double length() const { return max_x_ > min_x_ ? max_x_ - min_x_ : 0.0; }
    double height() const { return max_y_ > min_y_ ? max_y_ - min_y_ : 0.0; }

private:
    // Metadata / bounding box and shape list.
    std::string name_ = "geometry";
    std::vector<Shape2D> shapes_;

    // Bounding box supports drag normalization and debug output without recomputing geometry each query.
    double min_x_ = 1.0e300;
    double max_x_ = -1.0e300;
    double min_y_ = 1.0e300;
    double max_y_ = -1.0e300;

    // Primitive constructors used internally while assembling geometry.
    void add_polygon(std::string name, std::vector<Point2> pts);
    void add_circle(std::string name, double cx, double cy, double r);
    void add_rect(std::string name, double xmin, double ymin, double xmax, double ymax);

    // Update stored bounding box with a newly added primitive.
    void update_bbox(const Shape2D& s);
};
