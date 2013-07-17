#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void eye_paths_fill_ray_buffer(
    starpu_data_handle_t eye_paths,
    starpu_data_handle_t eye_paths_indexes,
    starpu_data_handle_t hit_points_info,
    starpu_data_handle_t ray_buffer,
    const unsigned start,
    const unsigned end,
    const unsigned max_eye_path_depth) {

  codelets::starpu_eye_paths_fill_ray_buffer_args args = { start, end, max_eye_path_depth };

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::eye_paths_fill_ray_buffer;
  task->handles[0] = eye_paths;
  task->handles[1] = eye_paths_indexes;
  task->handles[2] = hit_points_info;
  task->handles[3] = ray_buffer;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }
