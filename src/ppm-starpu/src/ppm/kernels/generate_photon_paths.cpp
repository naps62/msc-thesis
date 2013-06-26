#include "ppm/kernels/codelets.h"
#include "ppm/kernels/generate_photon_paths.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void generate_photon_paths(
    RayBuffer&       ray_hit_buffer,
    vector<Seed>&    seed_buffer,
    const Config*    config,
    PtrFreeScene*    scene) {

  // kernel args
  struct args_generate_photon_paths args = { config, scene };

  // handles
  // ray_buffer
  starpu_data_handle_t handle_ray_buffer;
  starpu_vector_data_register(&handle_ray_buffer, 0, (uintptr_t)ray_hit_buffer.GetRayBuffer(), ray_hit_buffer.GetSize(), sizeof(Ray));
  // seed_buffer
  starpu_data_handle_t handle_seed_buffer;
  starpu_vector_data_register(&handle_seed_buffer, 0, (uintptr_t)&seed_buffer[0], seed_buffer.size(), sizeof(Seed));

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::generate_photon_paths;
  task->handles[0] = handle_ray_buffer;
  task->handles[1] = handle_seed_buffer;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }
