#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;

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
      PhotonPath* const photon_paths, const unsigned photon_paths_count,
      Seed* const seed_buffer,        // const unsigned seed_buffer_count,
      const PtrFreeScene* scene,
      const unsigned num_threads) {

    #pragma omp parallel for num_threads(num_threads)
    for(unsigned i = 0; i < photon_paths_count; ++i) {
      PhotonPath& path = photon_paths[i];
      Ray& ray = path.ray;
      float light_pdf;
      float pdf;
      Spectrum f;

      const float u0 = floatRNG(seed_buffer[i]);
      const float u1 = floatRNG(seed_buffer[i]);
      const float u2 = floatRNG(seed_buffer[i]);
      const float u3 = floatRNG(seed_buffer[i]);
      const float u4 = floatRNG(seed_buffer[i]);


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
  const timeval start_time = my_WallClockTime();

    const starpu_args args;
    unsigned iteration;
    starpu_codelet_unpack_args(args_orig, &args, &iteration);

    // buffers
    // photon paths
    PhotonPath* const photon_paths = reinterpret_cast<PhotonPath* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
    const unsigned photon_paths_count = STARPU_VECTOR_GET_NX(buffers[0]);
    // seeds
    Seed* const seed_buffer          = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
    //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[1]);

    generate_photon_paths_impl(photon_paths, photon_paths_count,
                               seed_buffer,  // seed_buffer_count,
                               args.cpu_scene,
                               starpu_combined_worker_get_size());

  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, starpu_combined_worker_get_size(), iteration, start_time, end_time, "(6) generate_photon_paths");

  }

} } }
