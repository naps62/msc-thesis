#include "ppm/kernels/kernels.h"

#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
#include "ppm/kernels/helpers.cuh"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cpu {

  void generate_photon_paths_impl(
      Ray* const rays,                const unsigned rays_count,
      PhotonPath* const photon_paths, // const unsigned photon_paths_count,
      Seed* const seed_buffer,        // const unsigned seed_buffer_count,
      const Config* config,
      const PtrFreeScene* scene) {

    #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
    for(unsigned i = 0; i < rays_count; ++i) {
      Ray& ray = rays[i];
      PhotonPath& path = photon_paths[i];
      float light_pdf;
      float pdf;
      Spectrum f;

      const float u0 = floatRNG(seed_buffer[i]);
      const float u1 = floatRNG(seed_buffer[i]);
      const float u2 = floatRNG(seed_buffer[i]);
      const float u3 = floatRNG(seed_buffer[i]);
      const float u4 = floatRNG(seed_buffer[i]);


//std::cout << u0 << " " << u1 << " " << u2 << " " << u3 << " " << u4 << '\n';
      int light_index;
      ppm::LightType light_type;
      light_type = helpers::sample_all_lights(u0, scene->area_lights_count, scene->infinite_light, scene->sun_light, scene->sky_light, light_pdf, light_index);

      if (light_type == ppm::LIGHT_IL_IS) {
        helpers::infinite_light_sample_l(u1, u2, u3, u4, scene->infinite_light, scene->infinite_light_map, scene->bsphere, pdf, ray, path.flux);
      }
      else if (light_type == ppm::LIGHT_SUN)
        helpers::sun_light_sample_l(u1, u2, u3, u4, scene->sun_light, scene->bsphere, pdf, ray, path.flux);
      else if (light_type == ppm::LIGHT_IL_SKY)
        helpers::sky_light_sample_l(u1, u2, u3, u4, scene->sky_light, scene->bsphere, pdf, ray, path.flux);
      else
        helpers::triangle_light_sample_l(u1, u2, u3, u4, scene->area_lights[light_index], scene->mesh_descs, scene->colors, pdf, ray, path.flux);

      path.flux /= pdf * light_pdf;
      path.depth = 0;
      path.done = 0;
    }
  }


  void generate_photon_paths(void* buffers[], void* args_orig) {
    const starpu_args*  args   = (const starpu_args*) args_orig;
    const Config*       config = static_cast<const Config*>(args->cpu_config);
    const PtrFreeScene* scene  = static_cast<const PtrFreeScene*>(args->cpu_scene);

    // buffers
    // rays
    Ray* const rays           = reinterpret_cast<Ray* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
    const unsigned rays_count = STARPU_VECTOR_GET_NX(buffers[0]);
    // photon paths
    PhotonPath* const photon_paths = reinterpret_cast<PhotonPath* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
    // const unsigned photon_paths_count = STARPU_VECTOR_GET_NX(buffers[1]);
    // seeds
    Seed* const seed_buffer          = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[2]));
    //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[2]);


    generate_photon_paths_impl(rays,         rays_count,
                               photon_paths, // photon_paths_count
                               seed_buffer,  // seed_buffer_count,
                               config,
                               scene);


  }

} } }
