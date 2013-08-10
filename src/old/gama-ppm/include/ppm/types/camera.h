#ifndef _PPM_TYPES_CAMERA_H_
#define _PPM_TYPES_CAMERA_H_

#include <cstring>
#include "luxrays/utils/sdl/camera.h"
#include <ostream>
using std::ostream;

namespace ppm {

struct Camera {
	float lens_radius;
	float focal_distance;
	float yon, hither;

	float raster_to_camera_matrix[4][4];
	float camera_to_world_matrix[4][4];

	// recompiles camera coords
	void compile(luxrays::PerspectiveCamera& original) {
		yon = original.clipYon;
		hither = original.clipHither;
		lens_radius = original.lensRadius;
		focal_distance = original.focalDistance;

		memcpy(&raster_to_camera_matrix[0][0], original.GetRasterToCameraMatrix().m, sizeof(float) * 4 * 4);
		memcpy(&camera_to_world_matrix[0][0],  original.GetCameraToWorldMatrix().m,  sizeof(float) * 4 * 4);

	}
};

ostream& operator<< (ostream& os, const Camera& c);

}

#endif // _PPM_TYPES_CAMERA_H_
