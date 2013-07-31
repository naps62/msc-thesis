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

#ifndef _LUXRAYS_NORMAL_H
#define _LUXRAYS_NORMAL_H

#include "luxrays/core/geometry/vector.h"

class Normal {
public:
	// Normal Methods
	__H_D__ Normal(float _x = 0, float _y = 0, float _z = 0) :
		x(_x), y(_y), z(_z) {
	}
	__H_D__
	Normal operator-() const {
		return Normal(-x, -y, -z);
	}
	__H_D__
	Normal operator+(const Normal &v) const {
		return Normal(x + v.x, y + v.y, z + v.z);
	}
	__H_D__
	Normal & operator+=(const Normal &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	__H_D__
	Normal operator-(const Normal &v) const {
		return Normal(x - v.x, y - v.y, z - v.z);
	}
	__H_D__
	Normal & operator-=(const Normal &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	__H_D__
	Normal operator*(float f) const {
		return Normal(f * x, f * y, f * z);
	}
	__H_D__
	Normal & operator*=(float f) {
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}
	__H_D__
	Normal operator/(float f) const {
		float inv = 1.f / f;
		return Normal(x * inv, y * inv, z * inv);
	}
	__H_D__
	Normal & operator/=(float f) {
		float inv = 1.f / f;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}
	__H_D__
	float LengthSquared() const {
		return x * x + y * y + z * z;
	}
	__H_D__
	float Length() const {
		return sqrtf(LengthSquared());
	}
	__HD__
	explicit Normal(const Vector &v) :
		x(v.x), y(v.y), z(v.z) {
	}
	__H_D__
	float operator[](int i) const {
		return (&x)[i];
	}
	__H_D__
	float &operator[](int i) {
		return (&x)[i];
	}
	// Normal Public Data
	float x, y, z;
} ;
__H_D__
inline Normal operator*(float f, const Normal &n) {
	return Normal(f * n.x, f * n.y, f * n.z);
}
__H_D__
inline Vector::Vector(const Normal &n) :
	x(n.x), y(n.y), z(n.z) {
}

inline std::ostream &operator<<(std::ostream &os, const Normal &v) {
	os << "Normal[" << v.x << ", " << v.y << ", " << v.z << "]";
	return os;
}
__H_D__
inline Normal Normalize(const Normal &n) {
	return n / n.Length();
}
__H_D__
inline float Dot(const Normal &n1, const Normal &n2) {
	return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}
__H_D__
inline float AbsDot(const Normal &n1, const Normal &n2) {
	return fabsf(n1.x * n2.x + n1.y * n2.y + n1.z * n2.z);
}

#endif 	/* _LUXRAYS_NORMAL_H */
