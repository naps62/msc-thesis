/*
 * point.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_POINT_H_
#define _PPM_GEOMETRY_POINT_H_

#include <gama.h>
#include "ppm/geometry/vector.h"

namespace ppm {

struct Point {
	float x, y, z;

	/*
	 * constructors
	 */

	// default constructor
	__HYBRID__ Point(float _x = 0.f, float _y = 0.f, float _z = 0.f)
	: x(_x), y(_y), z(_z) { }

	// copy constructor
	__HYBRID__ Point(const Point& p)
	: x(p.x), y(p.y), z(p.z) { }

	// constructor from an array
	__HYBRID__ Point(float v[3])
	: x(v[0]), y(v[1]), z(v[2]) { }

	/*
	 * operators
	 */
	__HYBRID__ Point operator+ (const Point& p) const {
		return Point(x + p.x, y + p.y, z + p.z);
	}

	__HYBRID__ Point operator- (const Point& p) const {
		return Point(x - p.x, y - p.y, z - p.z);
	}

	__HYBRID__ Point& operator += (const Point& p) {
		x += p.x; y += p.y; z += p.z;
		return *this;
	}

	__HYBRID__ Point& operator -= (const Point& p) {
		x -= p.x; y -= p.y; z -= p.z;
		return *this;
	}

	__HYBRID__ Point operator* (const float f) const {
		return Point(f * x, f * y, f * z);
	}

	__HYBRID__ Point& operator*= (const float f) {
		x *= f; y *= f; z *= f;
		return *this;
	}

	__HYBRID__ Point operator/ (const float f) const {
		float inv = 1.f / f;
		return Point(inv * x, inv * y, inv * z);
	}

	__HYBRID__ Point& operator/= (const float f) {
		float inv = 1.f / f;
		x *= inv; y *= inv; z *= inv;
		return *this;
	}

	__HYBRID__ float operator[](const int i) const {
		return (&x)[i];
	}

	__HYBRID__ float& operator[](const int i) {
		return (&x)[i];
	}

	__HYBRID__ float distance(const Point& p) {
		return sqrt(distance_squared(p));
	}

	__HYBRID__ float distance_squared(const Point& p) {
		const Point diff = *this - p;
		return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
	}
};

__HYBRID__ __forceinline ostream& operator<<(ostream& os, const Point& p) {
	return os << "Point[" << p.x << ", " << p.y << ", " << p.z << "]";
}

__HYBRID__ __forceinline Point operator*(const float f, const Point &p) {
	return p * f;
}

}

#endif // _PPM_GEOMETRY_POINT_H_
