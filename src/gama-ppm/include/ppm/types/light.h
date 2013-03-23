/*
 * light.h
 *
 *  Created on: Mar 20, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_TYPES_LIGHT_H_
#define _PPM_TYPES_LIGHT_H_

#include "ppm/geometry/point.h"
#include "ppm/geometry/normal.h"
#include "ppm/pixel/spectrum.h"
#include "ppm/types/material.h"
#include "ppm/types.h"

namespace ppm {

struct TriangleLight {
	Point v0, v1, v2;
	Normal normal;
	float area;
	Spectrum gain;
	uint mesh_index, tri_index;
};

ostream& operator<< (ostream& os, const TriangleLight& l) {
	return os << "TriangleLight[" << l.v0 << ", " << l.v1 << ", " << l.v2 << ", "
			  << l.normal << ", " << l.area << ", " << l.gain << ", " << l.mesh_index << ", " << l.tri_index << "]";
}

struct InfiniteLight {
	bool exists;
	float shiftU, shiftV;
	Spectrum gain;
	uint width, height;
};

ostream& operator<< (ostream& os, const InfiniteLight& l) {
	return os << "InfiniteLight[" << l.exists << "; " << l.shiftU << "; " << l.shiftV << ", " << l.gain
			  << ", " << l.width << ", " << l.height << "]";
}

struct SunLight {
	bool exists;
	Vector dir;
	Spectrum gain;
	float turbidity;
	float rel_size;
	// xy vectors for cone sampling
	Vector x, y;
	float cos_theta_max;
	Spectrum color;
};

ostream& operator<< (ostream& os, const SunLight& l) {
	return os << "SunLight[" << l.exists << ", " << l.dir << ", " << l.gain
			  << ", " << l.turbidity << ", " << l.rel_size << ", " << l.x << ", " << l.y
			  << ", " << l.cos_theta_max << ", " << l.color << "]";
}

struct SkyLight {
	bool exists;
	Spectrum gain;
	float theta_s;
	float phi_s;
	float zenith_Y, zenith_x, zenith_y;
	float perez_Y[6], perez_x[6], perez_y[6];
};

ostream& operator<< (ostream& os, const SkyLight& l) {
	return os << "SkyLight[" << l.exists << ", " << l.gain << ", " << l.theta_s << ", "
			  << l.phi_s << "; " << l.zenith_Y << ", " << l.zenith_x << "; " << l.zenith_Y << ", "
			  << "perez_Y["
			      << l.perez_Y[0] << ' ' << l.perez_Y[1] << ' ' << l.perez_Y[2] << ' ' << l.perez_Y[3] << ' ' << l.perez_Y[4] << ' ' << l.perez_Y[5] << "], "
			  << "perez_Y["
			      << l.perez_x[0] << ' ' << l.perez_x[1] << ' ' << l.perez_x[2] << ' ' << l.perez_x[3] << ' ' << l.perez_x[4] << ' ' << l.perez_x[5] << "], "
			  << "perez_Y["
			      << l.perez_y[0] << ' ' << l.perez_y[1] << ' ' << l.perez_y[2] << ' ' << l.perez_y[3] << ' ' << l.perez_y[4] << ' ' << l.perez_y[5] << "] "
			  << "]";
}

}

#endif // _PPM_TYPES_LIGHT_H_
