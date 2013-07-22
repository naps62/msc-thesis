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
    const HitPointStaticInfo* const hit_points_info,
    HitPoint* const hit_points,
    const unsigned size,
    const float alpha,
    const unsigned photons_traced) {

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned int i = 0; i < size; ++i) {
    const HitPointStaticInfo& hpi = hit_points_info[i];
    HitPoint& hp = hit_points[i];

    switch (hpi.type) {
      case CONSTANT_COLOR:
        hp.radiance = hpi.throughput;
        break;
      case SURFACE:
        if (hp.accum_photon_count > 0) {
          const unsigned long long pcount = hp.photon_count + hp.accum_photon_count;
          const float g = alpha * pcount / (hp.photon_count * alpha + hp.accum_photon_count);

          hp.accum_photon_radius2 *= g;
          hp.reflected_flux = (hp.reflected_flux + hp.accum_reflected_flux) * g;
          hp.photon_count = pcount;

          const double k = 1.0 / (M_PI * hp.accum_photon_radius2 * photons_traced);

          hp.radiance = hp.reflected_flux * k;
          hp.accum_photon_count = 0;
          hp.accum_reflected_flux = Spectrum();
        }
        break;
      default:
        assert(false);
    }
  }
}


void accum_flux(void* buffers[], void* args_orig) {

  // cl_args
  const starpu_args*  args   = (const starpu_args*) args_orig;
  const Config* config = args->cpu_config;

  // buffers
  const HitPointStaticInfo* const hit_points_info = reinterpret_cast<const HitPointStaticInfo* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
        HitPoint*           const hit_points      = reinterpret_cast<      HitPoint*           const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  accum_flux_impl(hit_points_info, hit_points, size, config->alpha, config->engine_chunk_size);
}

} } }
