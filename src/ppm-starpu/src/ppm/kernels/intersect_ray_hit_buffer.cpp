#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void intersect_ray_hit_buffer(
    RayBuffer&    ray_hit_buffer) {

  // ray buffer
  starpu_data_handle_t handle_rays;
  starpu_vector_data_register(&handle_rays, 0, (uintptr_t)ray_hit_buffer.GetRayBuffer(), ray_hit_buffer.GetRayCount(), sizeof(Ray));
  // hit buffer
  starpu_data_handle_t handle_hits;
  starpu_vector_data_register(&handle_hits, 0, (uintptr_t)ray_hit_buffer.GetHitBuffer(), ray_hit_buffer.GetRayCount(), sizeof(RayHit));

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::intersect_ray_hit_buffer;
  task->handles[0] = handle_rays;
  task->handles[1] = handle_hits;
  task->cl_arg      = &codelets::generic_args;
  task->cl_arg_size = sizeof(codelets::generic_args);

  // submit
  starpu_task_submit(task);
}

} }
