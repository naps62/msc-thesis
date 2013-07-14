#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void advance_photon_paths(
  starpu_data_handle_t ray_buffer,
  starpu_data_handle_t hit_buffer,
  starpu_data_handle_t photon_paths,
  starpu_data_handle_t hit_points_info,
  starpu_data_handle_t hit_points,
  starpu_data_handle_t seed_buffer) {

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::advance_photon_paths;
  task->handles[0] = ray_buffer;
  task->handles[1] = hit_buffer;
  task->handles[2] = photon_paths;
  task->handles[3] = hit_points_info;
  task->handles[4] = hit_points;
  task->handles[5] = seed_buffer;
  task->cl_arg      = &codelets::generic_args;
  task->cl_arg_size = sizeof(codelets::generic_args);

  // submit
  starpu_task_submit(task);
}

} }
