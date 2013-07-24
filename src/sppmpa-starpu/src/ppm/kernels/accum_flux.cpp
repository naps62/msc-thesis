#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void accum_flux(
  starpu_data_handle_t hit_points_info,
  starpu_data_handle_t hit_points,
  const unsigned photons_traced) {

  codelets::starpu_accum_flux_args args = {
    codelets::generic_args.cpu_config,
    codelets::generic_args.gpu_config,
    photons_traced
  };

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::accum_flux;
  task->handles[0]  = hit_points_info;
  task->handles[1]  = hit_points;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }
