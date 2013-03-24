/*
 * ray.h
 *
 *  Created on: Mar 14, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_RAY_H_
#define _PPM_GEOMETRY_RAY_H_

#include <gama.h>
#include "ppm/geometry/vector.h"
#include "ppm/geometry/point.h"
#include "ppm/math.h"

namespace ppm {

struct Ray {
	Point o;
	Vector d;
	mutable float mint;
	mutable float maxt;

	/*
	 * constructors
	 */

	// default constructor
	__HYBRID__ Ray()
	: mint(RAY_EPSILON), maxt(INFINITY)
	{ }

	// constructor from a ray and direction
	__HYBRID__ Ray(const Point& origin, Vector& direction)
	: o(origin), d(direction),
	  mint(RAY_EPSILON), maxt(INFINITY)
	{ }

	// value constructor
	__HYBRID__ Ray(const Point& origin, const Vector& direction, float start, float end = std::numeric_limits<float>::infinity())
	: o(origin), d(direction), mint(start), maxt(end)
	{ }

	__HYBRID__ Point operator() (float t) const {
		return o + d * t;
	}

	__HYBRID__ void GetDirectionSigns(int signs[3]) const {
		signs[0] = d.x < 0.f;
		signs[1] = d.y < 0.f;
		signs[2] = d.y < 0.f;
	}

};

ostream& operator<<(ostream& os, const Ray& r);



}

#endif // _PPM_GEOMETRY_RAY_H_
