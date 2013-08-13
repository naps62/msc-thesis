#ifndef _PPM_TYPES_HITPOINTS_H_
#define _PPM_TYPES_HITPOINTS_H_

#include "utils/common.h"
#include <beast/common/types.hpp>

namespace ppm {

enum HitPointRadianceType {
  SURFACE,
  CONSTANT_COLOR
};

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
  Spectrum reflected_flux;
  Spectrum radiance;
};

}

#endif // _PPM_TYPES_HITPOINTS_H_
