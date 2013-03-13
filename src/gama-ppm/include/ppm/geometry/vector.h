/*
 * vector.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_VECTOR_H_
#define _PPM_GEOMETRY_VECTOR_H_

#include <gama.h>
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

	__HYBRID__ float dot(const Vector& v) const {
		return x * v.x + y * v.y + z * v.z;
	}

	__HYBRID__ float abs_dot(const Vector& v) const {
		return fabsf(this->dot(v));
	}

	__HYBRID__ Vector cross(const Vector &v) const {
		return Vector((y * v.z) - (z * v.y),
					(z * v.x) - (x * v.z),
					(x * v.y) - (y * v.x));
	}

	__HYBRID__ Vector normalize() const {
		return this / this->length();
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

	static __HYBRID__ void coordinate_system(const Vector& v1, Vector* v2, Vector* v3) {
		if (fabsf(v1.x) > fabsf(v1.y)) {
			float inv_len = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
			*v2 = Vector(-v1.z * inv_len, 0.f, v1.x * inv_len);
		} else {
			float inv_len = 1.f / sqrtf(v1.y * v1.y + v1.z * v1.z);
			*v2 = Vector(0.f, v1.z * inv_len, -v1.y * inv_len);
		}
		*v3 = v1.cross(*v2);
	}

	static __HYBRID__ Vector spherical_direction(float sintheta, float costheta, float phi) {
		return Vector(sintheta * cosf(phi), sintheta * sinf(phi), costheta);
	}

	static __HYBRID__ Vector spherical_direction(float sintheta, float costheta, float phi, const Vector& x, const Vector& y, const Vector& z) {
		return sintheta * cosf(phi) * x + sintheta * sinf(phi) * y + costheta * z;
	}
};

__HYBRID__ __forceinline ostream& operator<<(ostream& os, const Vector& v) {
	return os << "Vector[" << v.x << ", " << v.y << ", " << v.z << "]";
}

__HYBRID__ __forceinline Vector operator*(const float f, const Vector &v) {
	return v * f;
}


}

#endif // _PPM_GEOMETRY_VECTOR_H_
