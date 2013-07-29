#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void advance_photon_paths(
  starpu_data_handle_t photon_paths,
  starpu_data_handle_t hit_points_info,
  starpu_data_handle_t hit_points,
  starpu_data_handle_t seed_buffer,
  const float photon_radius2) {

  codelets::starpu_advance_photon_paths_args args = {
    codelets::generic_args.cpu_config,
    codelets::generic_args.cpu_scene,
    codelets::generic_args.cpu_hash_grid,
    codelets::generic_args.gpu_config,
    codelets::generic_args.gpu_scene,
    codelets::generic_args.gpu_hash_grid,
    photon_radius2
  };

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::advance_photon_paths;
  task->handles[0] = photon_paths;
  task->handles[1] = hit_points_info;
  task->handles[2] = hit_points;
  task->handles[3] = seed_buffer;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }
