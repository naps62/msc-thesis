#include "ppm/kernels/generate_eye_paths.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void k_cpu_generate_eye_paths(void *buffers[], void *args);

starpu_codelet cl_generate_eye_paths = {
  .where = STARPU_CPU,
  .type = STARPU_FORKJOIN,
  .cpu_funcs = { k_cpu_generate_eye_paths, NULL },
  .nbuffers = 2,
  .modes = { STARPU_W, STARPU_RW }
};

void generate_eye_paths(
    vector<EyePath> eye_paths,
    vector<Seed>    seed_buffer,
    const Config*   config,
    PtrFreeScene*   scene) {

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
  task->cl = &cl_generate_eye_paths;
  task->handles[0] = handle_eye_paths;
  task->handles[1] = handle_seed_buffer;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}
