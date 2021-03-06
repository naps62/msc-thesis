#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;
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
    const float current_photon_radius2,
    const unsigned num_threads) {

  #pragma omp parallel for num_threads(num_threads)
  for(unsigned int i = 0; i < size; ++i) {
    const HitPointPosition& hpi = hit_points_info[i];
    HitPointRadiance& hp = hit_points[i];

    switch (hpi.type) {
      case CONSTANT_COLOR:
        hp.radiance = hpi.throughput;
        break;
      case SURFACE:
        break;
      default:
        assert(false);
    }

    const double k = 1.0 / (M_PI * current_photon_radius2 * photons_traced);
    hp.radiance = (hp.radiance + hp.reflected_flux * k);
  }
}


void accum_flux(void* buffers[], void* args_orig) {
  const timeval start_time = my_WallClockTime();

  // cl_args
  const starpu_args args;
  float alpha;
  unsigned photons_traced;
  unsigned iteration;
  starpu_codelet_unpack_args(args_orig, &args, &alpha, &photons_traced, &iteration);

  // buffers
  const HitPointPosition* const hit_points_info = reinterpret_cast<const HitPointPosition* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
        HitPointRadiance*           const hit_points      = reinterpret_cast<      HitPointRadiance*           const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  const float* const photon_radius2 = (const float* const)STARPU_VARIABLE_GET_PTR(buffers[2]);

  accum_flux_impl(hit_points_info,
                  hit_points,
                  size,
                  alpha,
                  photons_traced,
                  *photon_radius2,
                  starpu_combined_worker_get_size());

  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, starpu_combined_worker_get_size(), iteration, start_time, end_time, "(8) accum_flux");
}

} } }
