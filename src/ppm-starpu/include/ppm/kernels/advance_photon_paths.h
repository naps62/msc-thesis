#ifndef _PPM_KERNELS_ADVANCE_PHOTON_PATHS_H_
#define _PPM_KERNELS_ADVANCE_PHOTON_PATHS_H_

#include "utils/config.h"
#include "utils/random.h"
#include "ppm/ptrfreescene.h"
#include "ppm/types/paths.h"
#include <vector>
using std::vector;

namespace ppm { namespace kernels {

struct args_advance_photon_paths {
  const Config* config;
  PtrFreeScene* scene;
};

void advance_photon_paths (
  RayBuffer&          ray_hit_buffer,
  vector<PhotonPath>& photon_paths,
  vector<Seed>&       seed_buffer,
  const Config*       config,
  PtrFreeScene*       scene
);

} }

#endif // _PPM_KERNELS_ADVANCE_PHOTON_PATHS_H_
