#include "ppm/kernels/advance_eye_paths.h"

#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cpu {

void advance_eye_paths_impl(
    RayHit*   const hits,              const unsigned hits_count,
    EyePath*  const eye_paths,         const unsigned eye_paths_count,
    unsigned* const eye_paths_indexes, const unsigned eye_paths_indexes_count,
    const PtrFreeScene* const scene) {


  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned i = 0; i < hits_count; ++i) {
    EyePath& eye_path = eye_paths[eye_paths_indexes[i]];
    const RayHit& hit = hits[i];

    if (hit.Miss()) {
      // add a hit point
      HitPointStaticInfo& hp = scene->hit_points[eye_path.sampleIndex];
      hp.type = CONSTANT_COLOR;
      hp.scrX = eye_path.scrX;
      hp.scrY = eye_path.scrY;

      if (scene->infinite_light.exists || scene->sun_light.exists || ss->sky_light.exists) {
        if (scene->infinite_light.exists) {
          // scene->infinite_light_le
        }
        if (scene->sun_light.exists) {
          // scene->sun_light_le
        }
        if (scene->sky_light.exists) {
          // scene-<sky_light_le
        }
        hp.throughput *= eye_path.throughput;
      } else {
        hp.throughput = Spectrum();
        eye_path.done = true;
      }
    } else {
      // something was hit
      Point hit_point;
      Spectrum surface_color;
      Normal N, shade_N;
    }
  }
}


void advance_eye_paths(void* buffers[], void* args_orig) {

  // cl_args
  const args_advance_eye_paths* args = (args_advance_eye_paths*) args_orig;
  //const Config*       config = static_cast<const Config*>(args->config);
  const PtrFreeScene* scene  = static_cast<const PtrFreeScene*>(args->scene);

  // buffers
  // hit buffer
  RayHit* const hits = reinterpret_cast<RayHit* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned hits_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // eye paths
  EyePath* const eye_paths = reinterpret_cast<EyePath* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  const unsigned eye_paths_count = STARPU_VECTOR_GET_NX(buffers[1]);
  // eye paths indexes
  unsigned* const eye_paths_indexes = reinterpret_cast<unsigned* const>(STARPU_VECTOR_GET_PTR(buffers[2]));
  const unsigned eye_paths_indexes_count = STARPU_VECTOR_GET_NX(buffers[2]);


  advance_eye_paths_impl(hits,              hits_count,
                         eye_paths,         eye_paths_count,
                         eye_paths_indexes, eye_paths_indexes_count,
                         scene);
}

} } }
