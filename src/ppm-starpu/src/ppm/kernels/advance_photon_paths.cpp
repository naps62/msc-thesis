#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void advance_photon_paths(
    RayBuffer&          ray_hit_buffer,
    vector<PhotonPath>& photon_paths,
    vector<Seed>&       seed_buffer) {

  // handles
  // ray_buffer
  starpu_data_handle_t handle_ray_buffer;
  starpu_vector_data_register(&handle_ray_buffer, 0, (uintptr_t)ray_hit_buffer.GetRayBuffer(), ray_hit_buffer.GetSize(), sizeof(Ray));
  // hit_buffer
  starpu_data_handle_t handle_hit_buffer;
  starpu_vector_data_register(&handle_hit_buffer, 0, (uintptr_t)ray_hit_buffer.GetHitBuffer(), ray_hit_buffer.GetSize(), sizeof(RayHit));
  // photon_paths
  starpu_data_handle_t handle_photon_paths;
  starpu_vector_data_register(&handle_photon_paths, 0, (uintptr_t)&photon_paths[0], photon_paths.size(), sizeof(PhotonPath));
  // seed_buffer
  starpu_data_handle_t handle_seed_buffer;
  starpu_vector_data_register(&handle_seed_buffer, 0, (uintptr_t)&seed_buffer[0], seed_buffer.size(), sizeof(Seed));

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::advance_photon_paths;
  task->handles[0] = handle_ray_buffer;
  task->handles[1] = handle_hit_buffer;
  task->handles[2] = handle_photon_paths;
  task->handles[3] = handle_seed_buffer;
  task->cl_arg      = &codelets::generic_args;
  task->cl_arg_size = sizeof(codelets::generic_args);

  // submit
  starpu_task_submit(task);
}

} }
