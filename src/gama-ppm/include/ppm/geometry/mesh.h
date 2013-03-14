/*
 * ray.h
 *
 *  Created on: Mar 14, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_MESH_H_
#define _PPM_GEOMETRY_MESH_H_

#include <gama.h>

namespace ppm {

struct Mesh {
	uint vert_offset;
	uint tris_offset;
	uint colors_offset;
	bool has_colors;

	Matrix4x4 trans;
	Matrix4x4 inv_trans;
};

}

#endif // _PPM_GEOMETRY_MESH_H_
