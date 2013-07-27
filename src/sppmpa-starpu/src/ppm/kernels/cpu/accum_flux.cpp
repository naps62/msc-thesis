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

void accum_flux_impl(
    const HitPointPosition* const hit_points_info,
    HitPointRadiance* const hit_points,
    const unsigned size,
    const float alpha,
    const unsigned photons_traced,
    const float current_photon_radius2) {

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned int i = 0; i < size; ++i) {
    const HitPointPosition& hpi = hit_points_info[i];
    HitPointRadiance& hp = hit_points[i];

    hp.hits_count++;
    switch (hpi.type) {
      case CONSTANT_COLOR:
        hp.accum_radiance = hpi.throughput;
        break;
      case SURFACE:
        if (hp.accum_photon_count > 0) {
          hp.reflected_flux = hp.accum_reflected_flux;
          hp.accum_photon_count = 0;
          hp.accum_reflected_flux = Spectrum();
        }
        break;
      default:
        assert(false);
    }

    if (hp.hits_count > 0) {
      const double k = 1.0 / (M_PI * current_photon_radius2 * photons_traced);
      hp.radiance = (hp.accum_radiance + hp.reflected_flux * k);
    }
    hp.hits_count = 0;
    hp.accum_radiance = Spectrum();
  }
}


void accum_flux(void* buffers[], void* args_orig) {

  // cl_args
  const codelets::starpu_accum_flux_args*  args   = (const codelets::starpu_accum_flux_args*) args_orig;
  const Config* config = args->cpu_config;
  const unsigned photons_traced = args->photons_traced;
  const float current_photon_radius2 = args->current_photon_radius2;

  // buffers
  const HitPointPosition* const hit_points_info = reinterpret_cast<const HitPointPosition* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
        HitPointRadiance*           const hit_points      = reinterpret_cast<      HitPointRadiance*           const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  accum_flux_impl(hit_points_info, hit_points, size, config->alpha, photons_traced, current_photon_radius2);
}

} } }
