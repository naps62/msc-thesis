#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void advance_eye_paths(
    starpu_data_handle_t hit_points_info,
    starpu_data_handle_t eye_paths,
    starpu_data_handle_t seed_buffer) {

  starpu_insert_task(&codelets::advance_eye_paths,
    STARPU_RW, hit_points_info,
    STARPU_RW, eye_paths,
    STARPU_RW, seed_buffer,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    0
  );
  starpu_task_wait_for_all(); // TODO remove this from here later
}

} }
