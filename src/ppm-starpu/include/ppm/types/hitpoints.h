#ifndef _PPM_TYPES_HITPOINTS_H_
#define _PPM_TYPES_HITPOINTS_H_

#include "utils/common.h"
#include <beast/common/types.hpp>

namespace ppm {

enum HitPointType {
  SURFACE,
  CONSTANT_COLOR
};

struct HitPointStaticInfo {
  HitPointType type;
  float scr_x, scr_y;
  Spectrum throughput; // used for CONSTANT_COLOR and SURFACE types

  // used only SURFACE
  Point position;
  Vector wo;
  Normal normal;
  uint material_ss;
};

struct HitPoint {
  uint accum_photon_count;
  Spectrum accum_reflected_flux;
  Spectrum radiance;

  ulonglong photon_count;
  Spectrum reflected_flux;

  float accum_photon_radius; // TODO this should be only for PPM and SPPM

  uint constant_hits_count;
  uint surface_hits_count;
  Spectrum accum_radiance;
};

}

#endif // _PPM_TYPES_HITPOINTS_H_
