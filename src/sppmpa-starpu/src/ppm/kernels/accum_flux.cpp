#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void accum_flux(
  starpu_data_handle_t hit_points_info,
  starpu_data_handle_t hit_points,
  const unsigned photons_traced,
  const float current_photon_radius2) {

  starpu_insert_task(&codelets::accum_flux,
    STARPU_R,  hit_points_info,
    STARPU_RW, hit_points,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    STARPU_VALUE, &photons_traced,         sizeof(photons_traced),
    STARPU_VALUE, &current_photon_radius2,  sizeof(current_photon_radius2),
    0);
  starpu_task_wait_for_all(); // TODO remove this from here later
}

} }
