#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void intersect_ray_hit_buffer(
    starpu_data_handle_t ray_buffer,
    starpu_data_handle_t hit_buffer,
    const unsigned buffer_size) {

  codelets::starpu_intersect_ray_hit_buffer_args args = {
    codelets::generic_args.cpu_scene,
    codelets::generic_args.gpu_scene,
    buffer_size
  };

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::intersect_ray_hit_buffer;
  task->handles[0] = ray_buffer;
  task->handles[1] = hit_buffer;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }
