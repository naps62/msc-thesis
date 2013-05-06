#ifndef _PPM_GEOMETRY_VECTOR_H_
#define _PPM_GEOMETRY_VECTOR_H_

#include "utils/common.h"
#include "ppm/geometry/point.h"
#include "ppm/math.h"

namespace ppm {

struct Vector : Point {

  /*
   * constructors
   */

  // default constructor
  __HYBRID__ Vector(
      float x = 0.f,
      float y = 0.f,
      float z = 0.f)
  : Point(x, y, z) { }

  // copy constructor
  __HYBRID__ Vector(const Point& v)
  : Point(v) { }

  // copy from luxrays constructor
  Vector(const luxrays::Vector& v)
  : Point(v.x, v.y, v.z) { }

  // constructor from an array
  __HYBRID__ Vector(float v[3])
  : Point(v) { }

  /*
   * operators
   */
  __HYBRID__ Vector operator+ (const Point& p) const {
    return Vector(x + p.x, y + p.y, z + p.z);
  }

  __HYBRID__ Vector operator- (const Point& p) const {
    return Point(x - p.x, y - p.y, z - p.z);
  }

  __HYBRID__ Vector& operator += (const Point& p) {
    x += p.x; y += p.y; z += p.z;
    return *this;
  }

  __HYBRID__ Vector& operator -= (const Point& p) {
    x -= p.x; y -= p.y; z -= p.z;
    return *this;
  }

  __HYBRID__ Vector operator* (const float f) const {
    return Vector(f * x, f * y, f * z);
  }

  __HYBRID__ Vector& operator*= (const float f) {
    x *= f; y *= f; z *= f;
    return *this;
  }

  __HYBRID__ Vector operator/ (const float f) const {
    float inv = 1.f / f;
    return Vector(inv * x, inv * y, inv * z);
  }

  __HYBRID__ Vector& operator/= (const float f) {
    float inv = 1.f / f;
    x *= inv; y *= inv; z *= inv;
    return *this;
  }

  __HYBRID__ float length_squared() const {
    return x*x + y*y + z*z;
  }

  __HYBRID__ float length() const {
    return sqrt(length_squared());
  }

  __HYBRID__ Vector normalize() const {
    return *this / this->length();
  }

  __HYBRID__ float spherical_theta() const {
    return acosf(ppm::math::clamp(z, -1.f, 1.f));
  }

  __HYBRID__ float spherical_phi() {
    float p = atan2f(y, x);
    return (p < 0.f) ? p + 2.f * M_PI : p;
  }

  __HYBRID__ float cos_theta() {
    return z;
  }

  __HYBRID__ float sin_theta() {
    return sqrtf(ppm::math::max(0.f, 1.f - z * z));
  }

  __HYBRID__ float sin_theta2() {
    const float cos_theta = this->cos_theta();
    return 1.f - cos_theta * cos_theta;
  }

  __HYBRID__ float cos_phi() {
    return x / sin_theta();
  }

  __HYBRID__ float sin_phi() {
    return y / sin_theta();
  }

  __HYBRID__ bool same_hemisphere(const Vector& v) {
    return z * v.z > 0.f;
  }

  /*
   * dot, abs_dot and cross receive Points as arguments instead of Vector
   * This allows them to be more generic, and receive Normals as well instead of Vectors
   * while the calculations remain the same
   */

  __HYBRID__ static float dot(const Point& v1, const Point& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  }

  __HYBRID__ static float abs_dot(const Point& v1, const Point& v2) {
    return fabsf(dot(v1, v2));
  }

  __HYBRID__ static Vector cross(const Point& v1, const Point &v2) {
    return Vector((v1.y * v2.z) - (v1.z * v2.y),
          (v1.z * v2.x) - (v1.x * v2.z),
          (v1.x * v2.y) - (v1.y * v2.x));
  }

  __HYBRID__ static void coordinate_system(const Vector& v1, Vector* v2, Vector* v3) {
    if (fabsf(v1.x) > fabsf(v1.y)) {
      float inv_len = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
      *v2 = Vector(-v1.z * inv_len, 0.f, v1.x * inv_len);
    } else {
      float inv_len = 1.f / sqrtf(v1.y * v1.y + v1.z * v1.z);
      *v2 = Vector(0.f, v1.z * inv_len, -v1.y * inv_len);
    }
    *v3 = cross(v1, *v2);
  }

  __HYBRID__ static Vector spherical_direction(float sintheta, float costheta, float phi) {
    return Vector(sintheta * cosf(phi), sintheta * sinf(phi), costheta);
  }

  __HYBRID__ static Vector spherical_direction(float sintheta, float costheta, float phi, const Vector& x, const Vector& y, const Vector& z) {
    return sintheta * cosf(phi) * x + sintheta * sinf(phi) * y + costheta * z;
  }
};

ostream& operator<<(ostream& os, const Vector& v);

__HYBRID__ __forceinline Vector operator*(const float f, const Vector &v) {
  return v * f;
}

}

#endif // _PPM_GEOMETRY_VECTOR_H_
