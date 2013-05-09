#include "ppm/kernels/codelets.h"
#include "ppm/kernels/generate_eye_paths.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void generate_eye_paths(
    vector<EyePath>& eye_paths,
    vector<Seed>&    seed_buffer,
    const Config*    config,
    PtrFreeScene*    scene) {

  // kernel args
  struct args_generate_eye_paths args = { config, scene };

  // handles
  // eye_paths
  starpu_data_handle_t handle_eye_paths;
  starpu_vector_data_register(&handle_eye_paths, 0, (uintptr_t)&eye_paths[0], eye_paths.size(), sizeof(EyePath));
  // seed_buffer
  starpu_data_handle_t handle_seed_buffer;
  starpu_vector_data_register(&handle_seed_buffer, 0, (uintptr_t)&seed_buffer[0], seed_buffer.size(), sizeof(Seed));

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::generate_eye_paths;
  task->handles[0] = handle_eye_paths;
  task->handles[1] = handle_seed_buffer;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }
