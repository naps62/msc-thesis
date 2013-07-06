#ifndef _PPM_KERNELS_H_
#define _PPM_KERNELS_H_

#include "utils/config.h"
#include "utils/random.h"
#include "ppm/ptrfreescene.h"
#include "ppm/types/paths.h"
#include "ppm/kernels/codelets.h"
#include <vector>
using std::vector;
using ppm::kernels::codelets::starpu_args;

namespace ppm { namespace kernels {

void generate_eye_paths (
  vector<EyePath>& eye_paths,
  vector<Seed>&    seed_buffer
);

void generate_photon_paths (
  RayBuffer&          ray_hit_buffer,
  vector<PhotonPath>& photon_paths,
  vector<Seed>&       seed_buffer
);

void advance_eye_paths (
  vector<HitPointStaticInfo>& hit_points,
  RayBuffer&                  ray_hit_buffer,
  vector<EyePath>&            eye_paths,
  vector<unsigned>&           eye_paths_indexes,
  vector<Seed>&               seed_buffer
);

void advance_photon_paths (
  RayBuffer&          ray_hit_buffer,
  vector<PhotonPath>& photon_paths,
  vector<Seed>&       seed_buffer
);

void intersect_ray_hit_buffer (
  RayBuffer&    ray_hit_buffer
);

} }

#endif // _PPM_KERNELS_H_
