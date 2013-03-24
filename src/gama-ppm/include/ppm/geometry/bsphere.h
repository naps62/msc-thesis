/*
 * bsphere.h
 *
 *  Created on: Mar 14, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_BSPHERE_H_
#define _PPM_GEOMETRY_BSPHERE_H_

#include <gama.h>
#include "ppm/geometry/vector.h"
#include "ppm/geometry/point.h"
#include "ppm/geometry/ray.h"
#include "ppm/math.h"

namespace ppm {

struct BSphere {
	Point center;
	float radius;

	/*
	 * constructors
	 */

	__HYBRID__ BSphere()
	: center(0.f, 0.f, 0.f), radius(0.f)
	{ }

	__HYBRID__ BSphere(const Point& c, const float r)
	: center(c), radius(r)
	{ }
};

ostream& operator<<(ostream& os, const BSphere& s);

}

#endif // _PPM_GEOMETRY_BSPHERE_H_
