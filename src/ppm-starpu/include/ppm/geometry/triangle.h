#ifndef _PPM_GEOMETRY_TRIANGLE_H_
#define _PPM_GEOMETRY_TRIANGLE_H_

#include "utils/common.h"
#include "ppm/geometry/point.h"
#include "ppm/geometry/ray.h"
#include "ppm/geometry/rayhit.h"
#include "ppm/math.h"

namespace ppm {

struct Triangle {
  uint v[3];

  /*
   * constructors
   */

  // default constructor
  __HYBRID__ Triangle() { }

  // value constructor
  __HYBRID__ Triangle(const uint v0, const uint v1, const uint v2) {
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
  }

  // copy from luxrays constructor
  Triangle(const luxrays::Triangle& tri) {
    v[0] = tri.v[0];
    v[1] = tri.v[1];
    v[2] = tri.v[2];
  }

  __HYBRID__ BBox world_bound(const Point *verts) const {
    const Point& p0 = verts[v[0]];
    const Point& p1 = verts[v[1]];
    const Point& p2 = verts[v[2]];

    return BBox(p0, p1).join(p2);
  }

  __HYBRID__ bool intersect(const Ray& ray, const Point *verts, RayHit *triangle_hit) const {
    const Point& p0 = verts[v[0]];
    const Point& p1 = verts[v[1]];
    const Point& p2 = verts[v[2]];
    const Vector e1 = p1 - p0;
    const Vector e2 = p2 - p0;
    const Vector s1 = Vector::cross(ray.d, e2);

    const float divisor = Vector::dot(s1, e1);
    if (divisor == 0.f)
      return false;

    const float inv_divisor = 1.f / divisor;

    // compute first barycentric coordinate
    const Vector d = ray.o - p0;
    const float b1 = Vector::dot(d, s1) * inv_divisor;
    if (b1 < 0.f)
      return false;

    // compute second barycentric coordinate
    const Vector s2 = Vector::cross(d, e1);
    const float b2 = Vector::dot(ray.d, s2) * inv_divisor;
    if (b2 < 0.f)
      return false;

    const float b0 = 1.f - b1 - b2;
    if (b0 < 0.f)
      return false;

    // compute _t_ to intersection point
    const float t = Vector::dot(e2, s2) * inv_divisor;
    if (t < ray.mint || t > ray.maxt)
      return false;

    triangle_hit->t = t;
    triangle_hit->b1 = b1;
    triangle_hit->b2 = b2;
    return true;
  }

  __HYBRID__ float area(const Point *verts) const {
    const Point& p0 = verts[v[0]];
    const Point& p1 = verts[v[1]];
    const Point& p2 = verts[v[2]];

    return 0.5f * Vector::cross(p1 - p0, p2 - p0).length();
  }

  __HYBRID__ void sample(const Point *verts, const float u0, const float u1, Point *p, float *b0, float *b1, float *b2) const {
    Triangle::uniform_sample(u0, u1, b0, b1);

    // get triangle vertices in _p1_, _p2_ and _p3_
    const Point& p0 = verts[v[0]];
    const Point& p1 = verts[v[1]];
    const Point& p2 = verts[v[2]];

    *b2 = 1.f - (*b0) - (*b1);
    *p  = (*b0) * p0 + (*b1) * p1 + (*b2) * p2;
  }

  __HYBRID__ float area(const Point& p0, const Point& p1, const Point& p2) {
    return 0.5f * Vector::cross(p1 - p2, p2 - p0).length();
  }

  __HYBRID__ static void uniform_sample(const float u0, const float u1, float *u, float *v) {
    float su1 = sqrtf(u0);
    *u = 1.f - su1;
    *v = u1 * su1;
  }

};

ostream& operator<<(ostream& os, const Triangle& tri);



}

#endif // _PPM_GEOMETRY_TRIANGLE_H_
