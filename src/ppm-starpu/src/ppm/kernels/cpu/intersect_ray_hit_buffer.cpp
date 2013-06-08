#include "ppm/kernels/intersect_ray_hit_buffer.h"

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

void intersect_ray_hit_buffer_impl(
    Ray* const rays, const unsigned rays_count,
    RayHit* const hits,
    const PtrFreeScene* scene) {

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned int i = 0; i < rays_count; ++i) {
    hits[i].SetMiss();
    scene->intersect(rays[i], hits[i]);
  }
}


void intersect_ray_hit_buffer(void* buffers[], void* args_orig) {

  // cl_args
  const args_intersect_ray_hit_buffer* args = (args_intersect_ray_hit_buffer*) args_orig;
  //const Config*       config = static_cast<const Config*>(args->config);
  const PtrFreeScene* scene  = static_cast<const PtrFreeScene*>(args->scene);

  // buffers
  Ray* const rays = reinterpret_cast<Ray* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned rays_count = STARPU_VECTOR_GET_NX(buffers[0]);
  RayHit* const hits = reinterpret_cast<RayHit* const>(STARPU_VECTOR_GET_PTR(buffers[1]));


  intersect_ray_hit_buffer_impl(rays, rays_count, hits, scene);
}

} } }
