#ifndef _PPM_TYPES_HITPOINTS_H_
#define _PPM_TYPES_HITPOINTS_H_

#include "utils/common.h"
#include <beast/common/types.hpp>

namespace ppm {

enum HitPointRadianceType {
  SURFACE,
  CONSTANT_COLOR
};

/*struct HitPointStaticInfo {
  HitPointRadianceType type;
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

  float accum_photon_radius2;

  uint constant_hits_count;
  uint surface_hits_count;
  Spectrum accum_radiance;
};*/

struct HitPointPosition {
  HitPointRadianceType type;
  float scr_x, scr_y;

  // Used for CONSTANT_COLOR and SURFACE type
  Spectrum throughput;

  // Used for SURFACE type
  Point position;
  Vector wo;
  Normal normal;

  uint material_ss;
};

struct HitPointRadiance {
  unsigned int accum_photon_count;
  Spectrum accum_reflected_flux;
  Spectrum accum_radiance;
  Spectrum radiance;

  unsigned long long hits_count;
  unsigned long long photon_count;
  Spectrum reflected_flux;
};

}

#endif // _PPM_TYPES_HITPOINTS_H_
