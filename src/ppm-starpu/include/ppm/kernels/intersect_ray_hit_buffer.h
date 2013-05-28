#ifndef _PPM_KERNELS_EYE_PATHS_TO_HIT_POINTS_H_
#define _PPM_KERNELS_EYE_PATHS_TO_HIT_POINTS_H_

#include "utils/config.h"
#include "utils/random.h"
#include "ppm/ptrfreescene.h"
#include "ppm/types/paths.h"
#include <vector>
using std::vector;

namespace ppm { namespace kernels {

struct args_intersect_ray_hit_buffer {
  //const Config* config;
  PtrFreeScene* scene;
};

void intersect_ray_hit_buffer (
  RayBuffer&    ray_hit_buffer,
  //const Config* config,
  PtrFreeScene* scene
);

} }

#endif // _PPM_KERNELS_EYE_PATHS_TO_HIT_POINTS_H_
