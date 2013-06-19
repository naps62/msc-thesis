#include "ppm/kernels/codelets.h"
#include "ppm/kernels/advance_eye_paths.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void advance_eye_paths(
    vector<HitPointStaticInfo>& hit_points,
    RayBuffer&                  ray_hit_buffer,
    vector<EyePath>&            eye_paths,
    vector<unsigned>&           eye_paths_indexes,
    vector<Seed>&               seed_buffer,
    //const Config* config,
    PtrFreeScene* scene) {

  // kernel args
  struct args_advance_eye_paths args = { /*config,*/ scene };

  // hit points static info
  starpu_data_handle_t handle_hit_points;
  starpu_vector_data_register(&handle_hit_points, 0, (uintptr_t)&hit_points[0], hit_points.size(), sizeof(HitPointStaticInfo));
  // hit buffer
  starpu_data_handle_t handle_hits;
  starpu_vector_data_register(&handle_hits, 0, (uintptr_t)ray_hit_buffer.GetHitBuffer(), ray_hit_buffer.GetRayCount(), sizeof(RayHit));
  // eye paths
  starpu_data_handle_t handle_eye_paths;
  starpu_vector_data_register(&handle_eye_paths, 0, (uintptr_t)&eye_paths[0], eye_paths.size(), sizeof(EyePath));
  // eye paths indexes
  starpu_data_handle_t handle_indexes;
  starpu_vector_data_register(&handle_indexes, 0, (uintptr_t)&eye_paths_indexes[0], eye_paths_indexes.size(), sizeof(unsigned));
  // seed buffer
  starpu_data_handle_t handle_seeds;
  starpu_vector_data_register(&handle_seeds, 0, (uintptr_t)&seed_buffer[0], seed_buffer.size(), sizeof(Seed));

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::advance_eye_paths;
  task->handles[0] = handle_hit_points;
  task->handles[1] = handle_hits;
  task->handles[2] = handle_eye_paths;
  task->handles[3] = handle_indexes;
  task->handles[4] = handle_seeds;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);

  // submit
  starpu_task_submit(task);
}

} }
