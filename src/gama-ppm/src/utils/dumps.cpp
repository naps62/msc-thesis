/*
 * dumps.cpp
 *
 *  Created on: Mar 24, 2013
 *      Author: Miguel Palhas
 */

#include "ppm/types.h"

namespace ppm {

// Camera
ostream& operator<< (ostream& os, const Camera& c) {
	return os << "Camera[" << c.lens_radius << ", " << c.focal_distance << ", " << c.yon << ", " << c.hither
			  << "raster_to_camera[" << c.raster_to_camera_matrix[0] << ' ' << c.raster_to_camera_matrix[1] << ' ' << c.raster_to_camera_matrix[2] << ' ' << c.raster_to_camera_matrix[3] << "]"
			  << "camera_to_world["  << c.camera_to_world_matrix[0]  << ' ' << c.camera_to_world_matrix[1]  << ' ' << c.camera_to_world_matrix[2]  << ' ' << c.camera_to_world_matrix[3]  << "] ]";
}

// BBox
ostream& operator<<(ostream& os, const BBox& bbox) {
	return os << "BBox[" << bbox.pmin << ", " << bbox.pmax << "]";
}

// BSphere
ostream& operator<<(ostream& os, const BSphere& s) {
	return os << "BSphere[" << s.center << ", " << s.radius << "]";
}

// Matrix4x4
ostream & operator<<(ostream &os, const Matrix4x4 &mat) {
	os << "Matrix4x4[ ";
	for (int i = 0; i < 4; ++i) {
		os << "[ ";
		for (int j = 0; j < 4; ++j) {
			os << mat.m[i][j];
			if (j != 3) os << ", ";
		}
		os << " ] ";
	}
	return os << " ] ";
}

// Mesh
ostream& operator<< (ostream& os, const Mesh& m) {
	return os  << "Mesh["
		<< "verts_offset("  << m.verts_offset << "), "
		<< "tris_offset("   << m.tris_offset << "), "
		<< "colors_offset(" << m.colors_offset << "), "
		<< "has_colors("    << m.has_colors << "), "
		<< "trans:"         << m.trans << ", "
		<< "inv_trans:"     << m.inv_trans
		<< "]";
}

// Normal
ostream& operator<<(ostream& os, const Normal& n) {
	return os << "Normal[" << n.x << ", " << n.y << ", " << n.z << "]";
}

// Point
ostream& operator<<(ostream& os, const Point& p) {
	return os << "Point[" << p.x << ", " << p.y << ", " << p.z << "]";
}

// Ray
ostream& operator<<(ostream& os, const Ray& r) {
	return os << "Ray[" << r.o << ", " << r.d << ", " << r.mint << ", " << r.maxt << "]";
}

// RayHit
ostream& operator<< (ostream& os, const RayHit& r) {
	return os << "RayHit[" << r.t << ", " << r.b1 << ", " << r.b2 << ", " << r.index << "]";
}

// Triangle
ostream& operator<<(ostream& os, const Triangle& tri) {
	return os << "Triangle[" << tri.v[0] << ", " << tri.v[1] << ", " << tri.v[2] << "]";
}

// UV
ostream& operator<<(ostream& os, const UV& uv) {
	return os << "UV[" << uv.u << ", " << uv.v << "]";
}

ostream& operator<<(ostream& os, const Vector& v) {
	return os << "Vector[" << v.x << ", " << v.y << ", " << v.z << "]";
}

// TriangleLight
ostream& operator<< (ostream& os, const TriangleLight& l) {
	return os << "TriangleLight[" << l.v0 << ", " << l.v1 << ", " << l.v2 << ", "
			  << l.normal << ", " << l.area << ", " << l.gain << ", " << l.mesh_index << ", " << l.tri_index << "]";
}

// InfiniteLight
ostream& operator<< (ostream& os, const InfiniteLight& l) {
	return os << "InfiniteLight[" << l.exists << "; " << l.shiftU << "; " << l.shiftV << ", " << l.gain
			  << ", " << l.width << ", " << l.height << "]";
}

// SunLight
ostream& operator<< (ostream& os, const SunLight& l) {
	return os << "SunLight[" << l.exists << ", " << l.dir << ", " << l.gain
			  << ", " << l.turbidity << ", " << l.rel_size << ", " << l.x << ", " << l.y
			  << ", " << l.cos_theta_max << ", " << l.color << "]";
}

// SkyLight
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

// TexMap
ostream& operator<< (ostream& os, const TexMap& t) {
	return os << "TexMap[" << t.rgb_offset << "; " << t.alpha_offset << ", " << t.width << "; " << t.height << "]";
}

// Spectrum
ostream& operator<<(ostream& os, const Spectrum& s) {
	return os << "Spectrum[" << s.r << ", " << s.g << ", " << s.b << "]";
}

// Color
ostream& operator<< (ostream& os, const Color& c) {
	return os << "Color[" << c.r << "; " << c.g << ", " << c.b << "]";
}

// Material
ostream& operator<< (ostream& os, const Material& m) {
	os << "Material[" << m.type << ", " << m.diffuse << ", " << m.specular << ", ";
	switch (m.type) {
	case MAT_MATTE:       os << m.param.matte;        break;
	case MAT_AREALIGHT:   os << m.param.area_light;   break;
	case MAT_MIRROR:      os << m.param.mirror;       break;
	case MAT_GLASS:       os << m.param.glass;        break;
	case MAT_MATTEMIRROR: os << m.param.matte_mirror; break;
	case MAT_METAL:       os << m.param.metal;        break;
	case MAT_MATTEMETAL:  os << m.param.matte_metal;  break;
	case MAT_ALLOY:       os << m.param.alloy;        break;
	case MAT_ARCHGLASS:   os << m.param.arch_glass;   break;
	}
	return os << "]";
}

// MatteParam
ostream& operator<< (ostream& os, const MatteParam& m) {
	return os << "MatteParam[" << m.kd << "]";
}
// AreaLightParam
ostream& operator<< (ostream& os, const AreaLightParam& m) {
	return os << "AreaLightParam[" << m.gain << "]";
}
// MirrorParam
ostream& operator<< (ostream& os, const MirrorParam& m) {
	return os << "MirrorParam[" << m.kr << ", " << m.specular_bounce << "]";
}
// GlassParam
ostream& operator<< (ostream& os, const GlassParam& m) {
	return os << "GlassParam[" << m.refl << ", " << m.refrct << ", " << m.outside_ior << "; " << m.ior
			  << ", " << m.R0 << ", " << m.reflection_specular_bounce << ", " << m.transmission_specular_bounce << "]";
}
// MatteMirrorParam
ostream& operator<< (ostream& os, const MatteMirrorParam& m) {
	return os << "MatteMirrorParam[" << m.matte << ", " << m.mirror << ", "
			  << m.matte_filter << ", " << m.tot_filter << ", " << m.matte_pdf << ", " << m.mirror_pdf << "]";
}
// MetalParam
ostream& operator<< (ostream& os, const MetalParam& m) {
	return os << "MetalParam[" << m.kr << ", " << m.exp << ", " << m.specular_bounce << "]";
}
// MatteMetalParam
ostream& operator<< (ostream& os, const MatteMetalParam& m) {
	return os << "MatteMetalParam[" << m.matte << ", " << m.metal << ", "
			  << m.matte_filter << ", " << m.tot_filter << ", " << m.matte_pdf << ", " << m.metal_pdf << "]";
}
// AlloyParam
ostream& operator<< (ostream& os, const AlloyParam& m) {
	return os << "AlloyParam[" << m.diff << ", " << m.refl << ", " << m.exp << ", " << m.R0 << ", " << m.specular_bounce << "]";
}

// ArchGlassParam
ostream& operator<< (ostream& os, const ArchGlassParam& m) {
	return os << "ArchGlassParam[" << m.refl << ", " << m.refrct << ", "
			  << m.trans_filter << ", " << m.tot_filter << ", " << m.refl_pdf << ", " << m.trans_pdf << ", "
			  << m.reflection_specular_bounce << ", " << m.transmission_specular_bounce << "]";
}

}
