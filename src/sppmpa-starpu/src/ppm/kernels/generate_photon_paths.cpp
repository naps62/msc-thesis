#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void generate_photon_paths(
    starpu_data_handle_t ray_buffer,
    starpu_data_handle_t photon_paths,
    starpu_data_handle_t seed_buffer) {

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::generate_photon_paths;
  task->handles[0] = ray_buffer;
  task->handles[1] = photon_paths;
  task->handles[2] = seed_buffer;
  task->cl_arg      = &codelets::generic_args;
  task->cl_arg_size = sizeof(codelets::generic_args);

  // submit
  starpu_task_submit(task);
}

} }
