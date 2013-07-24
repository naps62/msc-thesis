#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void generate_eye_paths(
    starpu_data_handle_t eye_paths,
    starpu_data_handle_t seed_buffer) {

  // handles

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::generate_eye_paths;
  task->handles[0] = eye_paths;
  task->handles[1] = seed_buffer;
  task->cl_arg      = &codelets::generic_args;
  task->cl_arg_size = sizeof(codelets::generic_args);

  // submit
  starpu_task_submit(task);
}

} }
