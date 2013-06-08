#ifndef _PPM_KERNELS_ADVANCE_EYE_PATHS_H_
#define _PPM_KERNELS_ADVANCE_EYE_PATHS_H_

#include "utils/config.h"
#include "utils/random.h"
#include "ppm/ptrfreescene.h"
#include "ppm/types/paths.h"
#include <vector>
using std::vector;

namespace ppm { namespace kernels {

struct args_advance_eye_paths {
  //const Config* config;
  PtrFreeScene* scene;
};

void advance_eye_paths (
  RayBuffer&        ray_hit_buffer,
  vector<EyePath>&  eye_paths,
  vector<unsigned>& eye_paths_indexes,
  //const Config* config,
  PtrFreeScene* scene
);

} }

#endif // _PPM_KERNELS_ADVANCE_EYE_PATHS_H_
