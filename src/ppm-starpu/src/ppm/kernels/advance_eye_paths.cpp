#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void advance_eye_paths(
    starpu_data_handle_t hit_points_info,
    starpu_data_handle_t hit_buffer,
    starpu_data_handle_t eye_paths,
    starpu_data_handle_t eye_paths_indexes,
    starpu_data_handle_t seed_buffer,
    const unsigned buffer_size) {

  codelets::starpu_advance_eye_paths_args args = {
    codelets::generic_args.cpu_scene,
    codelets::generic_args.gpu_scene,
    codelets::generic_args.cpu_config,
    codelets::generic_args.gpu_config,
    buffer_size
  };

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::advance_eye_paths;
  task->handles[0] = hit_points_info;
  task->handles[1] = hit_buffer;
  task->handles[2] = eye_paths;
  task->handles[3] = eye_paths_indexes;
  task->handles[4] = seed_buffer;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }
