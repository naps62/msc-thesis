#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void advance_photon_paths(
  starpu_data_handle_t photon_paths,
  starpu_data_handle_t hit_points_info,
  starpu_data_handle_t hit_points,
  starpu_data_handle_t seed_buffer,
  const float photon_radius2) {

  starpu_insert_task(&codelets::advance_photon_paths,
    STARPU_RW, photon_paths,
    STARPU_R,  hit_points_info,
    STARPU_RW, hit_points,
    STARPU_RW, seed_buffer,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    STARPU_VALUE, &photon_radius2, sizeof(photon_radius2),
    0);
}

} }
