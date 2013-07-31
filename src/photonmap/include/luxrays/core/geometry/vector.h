/***************************************************************************
 *   Copyright (C) 1998-2010 by authors (see AUTHORS.txt )                 *
 *                                                                         *
 *   This file is part of LuxRays.                                         *
 *                                                                         *
 *   LuxRays is free software; you can redistribute it and/or modify       *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   LuxRays is distributed in the hope that it will be useful,            *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 *   LuxRays website: http://www.luxrender.net                             *
 ***************************************************************************/

#ifndef _LUXRAYS_VECTOR_H
#define _LUXRAYS_VECTOR_H

#include <cmath>
#include <ostream>
#include <functional>
#include <limits>
#include <algorithm>

#include "luxrays/core/utils.h"


class Point;
class Normal;

 class Vector {
public:
	// Vector Public Methods
	__H_D__
	Vector(float _x = 0.f, float _y = 0.f, float _z = 0.f) :
		x(_x), y(_y), z(_z) {
	}
	__H_D__
	explicit Vector(const Point &p);
	__H_D__
	Vector operator+(const Vector &v) const {
		return Vector(x + v.x, y + v.y, z + v.z);
	}
	__H_D__
	Vector & operator+=(const Vector &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	__H_D__
	Vector operator-(const Vector &v) const {
		return Vector(x - v.x, y - v.y, z - v.z);
	}
	__H_D__
	Vector & operator-=(const Vector &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	__H_D__
	bool operator==(const Vector &v) const {
		return x == v.x && y == v.y && z == v.z;
	}
	__H_D__
	Vector operator*(float f) const {
		return Vector(f*x, f*y, f * z);
	}
	__H_D__
	Vector & operator*=(float f) {
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}
	__H_D__
	Vector operator/(float f) const {
		float inv = 1.f / f;
		return Vector(x * inv, y * inv, z * inv);
	}
	__H_D__
	Vector & operator/=(float f) {
		float inv = 1.f / f;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}
	__H_D__
	Vector operator-() const {
		return Vector(-x, -y, -z);
	}
	__H_D__
	float operator[](int i) const {
		return (&x)[i];
	}
	__H_D__
	float &operator[](int i) {
		return (&x)[i];
	}
	__H_D__
	float LengthSquared() const {
		return x * x + y * y + z*z;
	}
	__H_D__
	float Length() const {
		return sqrtf(LengthSquared());
	}
	__H_D__

	explicit Vector(const Normal &n);

	// Vector Public Data
	float x, y, z;
} ;


inline std::ostream &operator<<(std::ostream &os, const Vector &v) {
	os << "Vector[" << v.x << ", " << v.y << ", " << v.z << "]";
	return os;
}
__HD__
inline Vector operator*(float f, const Vector &v) {
	return v*f;
}
__H_D__
inline float Dot(const Vector &v1, const Vector &v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
__HD__
inline float AbsDot(const Vector &v1, const Vector &v2) {
	return fabsf(Dot(v1, v2));
}
__HD__
inline Vector Cross(const Vector &v1, const Vector &v2) {
	return Vector((v1.y * v2.z) - (v1.z * v2.y),
			(v1.z * v2.x) - (v1.x * v2.z),
			(v1.x * v2.y) - (v1.y * v2.x));
}
__HD__
inline Vector Normalize(const Vector &v) {
	return v / v.Length();
}
__HD__
inline void CoordinateSystem(const Vector &v1, Vector *v2, Vector *v3) {
	if (fabsf(v1.x) > fabsf(v1.y)) {
		float invLen = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
		*v2 = Vector(-v1.z * invLen, 0.f, v1.x * invLen);
	} else {
		float invLen = 1.f / sqrtf(v1.y * v1.y + v1.z * v1.z);
		*v2 = Vector(0.f, v1.z * invLen, -v1.y * invLen);
	}
	*v3 = Cross(v1, *v2);
}
__HD__
inline Vector SphericalDirection(float sintheta, float costheta, float phi) {
	return Vector(sintheta * cosf(phi), sintheta * sinf(phi), costheta);
}
__HD__
inline Vector SphericalDirection(float sintheta, float costheta, float phi,
		const Vector &x, const Vector &y, const Vector &z) {
	return sintheta * cosf(phi) * x + sintheta * sinf(phi) * y +
			costheta * z;
}
__HD__
inline float SphericalTheta(const Vector &v) {
	return acosf(Clamp(v.z, -1.f, 1.f));
}
__HD__
inline float SphericalPhi(const Vector &v) {
	float p = atan2f(v.y, v.x);
	return (p < 0.f) ? p + 2.f * M_PI : p;
}
__HD__
inline float CosTheta(const Vector &w) {
	return w.z;
}
__HD__
inline float SinTheta(const Vector &w) {
	return sqrtf(Max(0.f, 1.f - w.z * w.z));
}
__HD__
inline float SinTheta2(const Vector &w) {
	return 1.f - CosTheta(w) * CosTheta(w);
}
__HD__
inline float CosPhi(const Vector &w) {
	return w.x / SinTheta(w);
}
__HD__
inline float SinPhi(const Vector &w) {
	return w.y / SinTheta(w);
}
__HD__
inline bool SameHemisphere(const Vector &w,
		const Vector &wp) {
	return w.z * wp.z > 0.f;
}



#endif	/* _LUXRAYS_VECTOR_H */
