/***************************************************************************
 *   Copyright (C) 1998-2009 by David Bucciarelli (davibu@interfree.it)    *
 *                                                                         *
 *   This file is part of SmallLuxGPU.                                     *
 *                                                                         *
 *   SmallLuxGPU is free software; you can redistribute it and/or modify   *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *  SmallLuxGPU is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 *   This project is based on PBRT ; see http://www.pbrt.org               *
 *   and Lux Renderer website : http://www.luxrender.net                   *
 ***************************************************************************/

#ifndef _LUXRAYS_TRIANGLE_H
#define	_LUXRAYS_TRIANGLE_H

#include "luxrays/core.h"

namespace luxrays {

__HD__
inline void UniformSampleTriangle(const float u0, const float u1, float *u, float *v) {
	float su1 = sqrtf(u0);
	*u = 1.f - su1;
	*v = u1 * su1;
}

class Triangle {
public:
	__HD__
	Triangle() { }
	__HD__
	Triangle(const unsigned int v0, const unsigned int v1, const unsigned int v2) {
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
	}
	__HD__
	BBox WorldBound(const Point *verts) const {
		const Point &p0 = verts[v[0]];
		const Point &p1 = verts[v[1]];
		const Point &p2 = verts[v[2]];

		return Union(BBox(p0, p1), p2);
	}
	__HD__
	bool Intersect(const Ray &ray, const Point *verts, RayHit *triangleHit) const {
		const Point &p0 = verts[v[0]];
		const Point &p1 = verts[v[1]];
		const Point &p2 = verts[v[2]];
		const Vector e1 = p1 - p0;
		const Vector e2 = p2 - p0;
		const Vector s1 = Cross(ray.d, e2);

		const float divisor = Dot(s1, e1);
		if (divisor == 0.f)
			return false;

		const float invDivisor = 1.f / divisor;

		// Compute first barycentric coordinate
		const Vector d = ray.o - p0;
		const float b1 = Dot(d, s1) * invDivisor;
		if (b1 < 0.f)
			return false;

		// Compute second barycentric coordinate
		const Vector s2 = Cross(d, e1);
		const float b2 = Dot(ray.d, s2) * invDivisor;
		if (b2 < 0.f)
			return false;

		const float b0 = 1.f - b1 - b2;
		if (b0 < 0.f)
			return false;

		// Compute _t_ to intersection point
		const float t = Dot(e2, s2) * invDivisor;
		if (t < ray.mint || t > ray.maxt)
			return false;

		triangleHit->t = t;
		triangleHit->b1 = b1;
		triangleHit->b2 = b2;

		return true;
	}
	__HD__
	float Area(const Point *verts) const {
		const Point &p0 = verts[v[0]];
		const Point &p1 = verts[v[1]];
		const Point &p2 = verts[v[2]];

		return 0.5f * Cross(p1 - p0, p2 - p0).Length();
	}
	__HD__
	void Sample(const Point *verts, const float u0,
		const float u1, Point *p, float *b0, float *b1, float *b2) const {
		UniformSampleTriangle(u0, u1, b0, b1);

		// Get triangle vertices in _p1_, _p2_, and _p3_
		const Point &p0 = verts[v[0]];
		const Point &p1 = verts[v[1]];
		const Point &p2 = verts[v[2]];
		*b2 = 1.f - (*b0) - (*b1);
		*p = (*b0) * p0 + (*b1) * p1 + (*b2) * p2;
	}
	__HD__
	static float Area(const Point &p0, const Point &p1, const Point &p2) {
		return 0.5f * Cross(p1 - p0, p2 - p0).Length();
	}

	unsigned int v[3];
};
__HD__
inline std::ostream & operator<<(std::ostream &os, const Triangle &tri) {
	os << "Triangle[" << tri.v[0] << ", " << tri.v[1] << ", " << tri.v[2] << "]";
	return os;
}

}

#endif	/* _LUXRAYS_TRIANGLE_H */
