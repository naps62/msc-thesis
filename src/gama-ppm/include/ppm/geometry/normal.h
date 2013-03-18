/*
 * normal.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_NORMAL_H_
#define _PPM_GEOMETRY_NORMAL_H_

#include <gama.h>
#include "ppm/geometry/point.h"
#include "ppm/math.h"

namespace ppm {

struct Normal : Point {

	/*
	 * constructors
	 */

	// default constructor
	__HYBRID__ Normal(
			float x = 0.f,
			float y = 0.f,
			float z = 0.f)
	: Point(x, y, z) { }

	// copy constructor
	__HYBRID__ Normal(const Point& v)
	: Point(v) { }

	// copy from luxrays constructor
	Normal(const luxrays::Normal& v)
	: Point(v.x, v.y, v.z) { }

	/*
	 * operators
	 */
	__HYBRID__ Normal operator-() const {
		return Normal(-x, -y, -z);
	}

	__HYBRID__ Normal operator+ (const Point& p) const {
		return Normal(x + p.x, y + p.y, z + p.z);
	}

	__HYBRID__ Normal operator- (const Point& p) const {
		return Normal(x - p.x, y - p.y, z - p.z);
	}

	__HYBRID__ Normal& operator += (const Point& p) {
		x += p.x; y += p.y; z += p.z;
		return *this;
	}

	__HYBRID__ Normal& operator -= (const Point& p) {
		x -= p.x; y -= p.y; z -= p.z;
		return *this;
	}

	__HYBRID__ Normal operator* (const float f) const {
		return Normal(f * x, f * y, f * z);
	}

	__HYBRID__ Normal& operator*= (const float f) {
		x *= f; y *= f; z *= f;
		return *this;
	}

	__HYBRID__ Normal operator/ (const float f) const {
		float inv = 1.f / f;
		return Normal(inv * x, inv * y, inv * z);
	}

	__HYBRID__ Normal& operator/= (const float f) {
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

	__HYBRID__ Normal normalize() const {
		return *this / this->length();
	}

	__HYBRID__ float dot(const Normal& n) const {
		return x * n.x + y * n.y + z * n.z;
	}

	__HYBRID__ float abs_dot(const Normal& n) const {
		return fabsf(x * n.x + y * n.y + z * n.z);
	}
};

__HYBRID__ __forceinline ostream& operator<<(ostream& os, const Normal& n) {
	return os << "Normal[" << n.x << ", " << n.y << ", " << n.z << "]";
}

__HYBRID__ __forceinline Vector operator*(const float f, const Normal &n) {
	return n * f;
}


}

#endif // _PPM_GEOMETRY_NORMAL_H_
