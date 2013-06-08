#include "ppm/kernels/codelets.h"
#include "ppm/kernels/advance_eye_paths.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void advance_eye_paths(
    RayBuffer&        ray_hit_buffer,
    vector<EyePath>&  eye_paths,
    vector<unsigned>& eye_paths_indexes,
    //const Config* config,
    PtrFreeScene* scene) {

  // kernel args
  struct args_advance_eye_paths args = { /*config,*/ scene };

  // hit buffer
  starpu_data_handle_t handle_hits;
  starpu_vector_data_register(&handle_hits, 0, (uintptr_t)ray_hit_buffer.GetHitBuffer(), ray_hit_buffer.GetRayCount(), sizeof(RayHit));
  // eye paths
  starpu_data_handle_t handle_eye_paths;
  starpu_vector_data_register(&handle_eye_paths, 0, (uintptr_t)&eye_paths[0], eye_paths.size(), sizeof(EyePath));
  // eye paths indexes
  starpu_data_handle_t handle_indexes;
  starpu_vector_data_register(&handle_indexes, 0, (uintptr_t)&eye_paths_indexes[0], eye_paths_indexes.size(), sizeof(unsigned));

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::advance_eye_paths;
  task->handles[0] = handle_hits;
  task->handles[1] = handle_eye_paths;
  task->handles[2] = handle_indexes;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }