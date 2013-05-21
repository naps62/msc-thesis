#include "ppm/types.h"

namespace ppm {

ostream& operator<< (ostream& os, const Color& c) {
  return os << "Color[" << c.r << ", " << c.b << ", " << c.b << "]";
}

// Camera
ostream& operator<< (ostream& os, const Camera& c) {
  os << "Camera[" << c.lens_radius << ", " << c.focal_distance << ", " << c.yon << ", " << c.hither << ", ";
  os << "raster_to_camera[";
  for(uint i = 0; i < 4; ++i)
    for(uint j = 0; j < 4; ++j)
      os << c.raster_to_camera_matrix[i][j] << ", ";
  os << "], camera_to_world[";
    for(uint i = 0; i < 4; ++i)
      for(uint j = 0; j < 4; ++j)
        os << c.camera_to_world_matrix[i][j] << ", ";
  return os << "] ]";
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

// TriangleLight
ostream& operator<< (ostream& os, const TriangleLight& l) {
  return os << "TriangleLight[" << l.v0 << ", " << l.v1 << ", " << l.v2 << ", "
        << l.normal << ", " << l.area << ", " << l.gain << ", " << l.mesh_index << ", " << l.tri_index << "]";
}

// InfiniteLight
ostream& operator<< (ostream& os, const InfiniteLight& l) {
  return os << "InfiniteLight[" << l.exists << ", " << l.shiftU << "; " << l.shiftV << ", " << l.gain
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
  return os << "TexMap[" << t.rgb_offset << ", " << t.alpha_offset << ", " << t.width << "; " << t.height << "]";
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
