#ifndef _PPM_KERNELS_BUILD_HIT_POINTS_H_
#define _PPM_KERNELS_BUILD_HIT_POINTS_H_

#include "utils/common.h"

struct args_build_hit_points {
  unsigned width;
  unsigned height;
  unsigned spp;
  unsigned hit_points;
};

_extern_c_ void k_build_hit_points(
  //void* scene,
  unsigned width,
  unsigned height,
  unsigned spp,
  unsigned hit_points
);

#endif // _PPM_KERNELS_BUILD_HIT_POINTS_H_
