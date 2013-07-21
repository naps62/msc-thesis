#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void accum_flux(
  starpu_data_handle_t hit_points_info,
  starpu_data_handle_t hit_points) {

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::accum_flux;
  task->handles[0]  = hit_points_info;
  task->handles[1]  = hit_points;
  task->cl_arg      = &codelets::generic_args;
  task->cl_arg_size = sizeof(codelets::generic_args);

  // submit
  starpu_task_submit(task);
}

} }
