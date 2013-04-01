#ifndef _PPM_GEOMETRY_MESH_H_
#define _PPM_GEOMETRY_MESH_H_

#include <gamalib/gamalib.h>
#include <ostream>
using std::ostream;

namespace ppm {
struct Mesh;
ostream& operator<< (ostream& os, const Mesh& m);
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
	  has_colors(false),
	  trans(),
	  inv_trans() { }

	Mesh(const Mesh& copy)
	: verts_offset(copy.verts_offset),
	  tris_offset(copy.tris_offset),
	  colors_offset(copy.colors_offset),
	  has_colors(copy.has_colors),
	  trans(copy.trans),
	  inv_trans(copy.inv_trans) { }
};

ostream& operator<< (ostream& os, const Mesh& m);

}

#endif // _PPM_GEOMETRY_MESH_H_
