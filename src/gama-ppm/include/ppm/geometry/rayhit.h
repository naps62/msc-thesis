/*
 * rayhit.h
 *
 *  Created on: Mar 14, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_RAYHIT_H_
#define _PPM_GEOMETRY_RAYHIT_H_

#include <gama.h>

namespace ppm {

struct RayHit {
	float t;
	float b1, b2; // barycentric coordinates of the hit point
	uint index;

	__HYBRID__ void set_miss() {
		index = 0xffffffffu;
	}

	__HYBRID__ bool is_miss() const {
		return index == 0xffffffffu;
	}
};

ostream& operator<< (ostream& os, const RayHit& r);

}

#endif // _PPM_GEOMETRY_RAYHIT_H_
