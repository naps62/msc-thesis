#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void generate_photon_paths(
    starpu_data_handle_t photon_paths,
    starpu_data_handle_t seed_buffer) {

  starpu_insert_task(&codelets::generate_photon_paths,
    STARPU_RW, photon_paths,
    STARPU_RW, seed_buffer,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    0
  );
}

} }
