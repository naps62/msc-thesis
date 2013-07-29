#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void generate_eye_paths(
    starpu_data_handle_t eye_paths,
    starpu_data_handle_t seed_buffer) {

  starpu_insert_task(&codelets::generate_eye_paths,
    STARPU_RW, eye_paths,
    STARPU_RW, seed_buffer,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    0
  );
}

} }
