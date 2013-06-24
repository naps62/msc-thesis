#ifndef _PPM_KERNELS_GENERATE_PHOTON_PATHS_H_
#define _PPM_KERNELS_GENERATE_PHOTON_PATHS_H_

#include "utils/config.h"
#include "utils/random.h"
#include "ppm/ptrfreescene.h"
#include "ppm/types/paths.h"
#include <vector>
using std::vector;

namespace ppm { namespace kernels {

struct args_generate_photon_paths {
  const Config* config;
  PtrFreeScene* scene;
};

void generate_photon_paths (
  RayBuffer& ray_hit_buffer,
  vector<Seed>&    seed_buffer,
  const Config*    config,
  PtrFreeScene*    scene
);

} }

#endif // _PPM_KERNELS_GENERATE_PHOTON_PATHS_H_
