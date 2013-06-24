#include "ppm/kernels/generate_photon_paths.h"

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

void generate_photon_paths_impl(
    Ray* const rays,         const unsigned rays_count,
    Seed* const seed_buffer, // const unsigned seed_buffer_count,
    const Config* config,
    const PtrFreeScene* scene) {

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned i = 0; i < rays_count; ++i) {
    float light_pdf;
    float pdf;

    const float u0 = floatRNG(seed_buffer[i]);
    const float u1 = floatRNG(seed_buffer[i]);
    const float u2 = floatRNG(seed_buffer[i]);
    const float u3 = floatRNG(seed_buffer[i]);
    const float u4 = floatRNG(seed_buffer[i]);
    const float u5 = floatRNG(seed_buffer[i]);

    int light_index;
    ppm::LightType;
    light_type = helpers::sample_all_lights(u0, scene->area_lights_count, light_pdf, light_index, scene->infinite_light, scene->sun_light, scene->sky_light, lpdf, light_index);

    if (light_type == ppm::LIGHT_IL_IS)
      helpers::
    else if (light_type == ppm::LIGHT_SUN)
      else if (light_type == ppm::LIGHT_IL_SKY)
    else
  }
}


void generate_photon_paths(void* buffers[], void* args_orig) {
  // cl_args
  const args_generate_photon_paths* args = (args_generate_photon_paths*) args_orig;
  const Config*       config = static_cast<const Config*>(args->config);
  const PtrFreeScene* scene  = static_cast<const PtrFreeScene*>(args->scene);

  // buffers
  // rays
  Ray* const rays           = reinterpret_cast<Ray* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned rays_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // seeds
  Seed* const seed_buffer          = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[1]);


  generate_photon_paths_impl(rays,        rays_count,
                          seed_buffer, // seed_buffer_count,
                          config,
                          scene);


}

} } }
