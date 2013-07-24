#include "ppm/kernels/kernels.h"

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
    Ray* const rays,
    RayHit* const hits,
    const unsigned buffer_size,
    const PtrFreeScene* scene) {

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned int i = 0; i < buffer_size; ++i) {
    hits[i].SetMiss();
    scene->intersect(rays[i], hits[i]);
  }
}


void intersect_ray_hit_buffer(void* buffers[], void* args_orig) {

  // cl_args
  const codelets::starpu_intersect_ray_hit_buffer_args* args = (const codelets::starpu_intersect_ray_hit_buffer_args*) args_orig;
  //const Config*       config = static_cast<const Config*>(args->cpu_config);

  // buffers
  Ray* const rays    = reinterpret_cast<Ray*    const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  RayHit* const hits = reinterpret_cast<RayHit* const>(STARPU_VECTOR_GET_PTR(buffers[1]));


  intersect_ray_hit_buffer_impl(rays, hits, args->buffer_size, args->cpu_scene);
}

} } }
