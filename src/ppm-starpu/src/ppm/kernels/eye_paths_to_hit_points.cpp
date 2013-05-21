#include "ppm/kernels/codelets.h"
#include "ppm/kernels/eye_paths_to_hit_points.h"

#include <starpu.h>

namespace ppm { namespace kernels {

void eye_paths_to_hit_points(
    vector<EyePath>&            eye_paths,
    vector<HitPointStaticInfo>& hit_points,
    vector<Seed>&               seed_buffer,
    const Config*               config,
    PtrFreeScene*    scene) {

  // kernel args
  struct args_eye_paths_to_hit_points args = { config, scene };

  // handles
  // eye_paths
  starpu_data_handle_t handle_eye_paths;
  starpu_vector_data_register(&handle_eye_paths, 0, (uintptr_t)&eye_paths[0], eye_paths.size(), sizeof(EyePath));
  // hit_points
  starpu_data_handle_t handle_hit_points;
  starpu_vector_data_register(&handle_hit_points, 0, (uintptr_t)&hit_points[0], hit_points.size(), sizeof(HitPointStaticInfo));

  // task definition
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &codelets::eye_paths_to_hit_points;
  task->handles[0] = handle_eye_paths;
  task->handles[1] = handle_hit_points;
  task->cl_arg      = &args;
  task->cl_arg_size = sizeof(args);


  // submit
  starpu_task_submit(task);
}

} }
