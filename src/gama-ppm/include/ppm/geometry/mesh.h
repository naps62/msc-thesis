/*
 * ray.h
 *
 *  Created on: Mar 14, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_GEOMETRY_MESH_H_
#define _PPM_GEOMETRY_MESH_H_

#include <gama.h>
#include <ostream>
using std::ostream;

namespace ppm {

struct Mesh {
	uint verts_offset;
	uint tris_offset;
	uint colors_offset;
	bool has_colors;

	Matrix4x4 trans;
	Matrix4x4 inv_trans;

	Mesh()
	: verts_offset(0),
	  tris_offset(0),
	  colors_offset(0),
	  trans(),
	  inv_trans() { }
};

ostream& operator<< (ostream& os, const Mesh& m);

}

#endif // _PPM_GEOMETRY_MESH_H_
