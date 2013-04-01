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


struct InfiniteLight {
	bool exists;
	float shiftU, shiftV;
	Spectrum gain;
	uint width, height;
};


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


struct SkyLight {
	bool exists;
	Spectrum gain;
	float theta_s;
	float phi_s;
	float zenith_Y, zenith_x, zenith_y;
	float perez_Y[6], perez_x[6], perez_y[6];
};

ostream& operator<< (ostream& os, const TriangleLight& l);
ostream& operator<< (ostream& os, const InfiniteLight& l);
ostream& operator<< (ostream& os, const SunLight& l);
ostream& operator<< (ostream& os, const SkyLight& l);

}

#endif // _PPM_TYPES_LIGHT_H_
