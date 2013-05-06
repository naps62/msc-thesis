#include "ppm/kernels/build_hit_points.h"

#include <starpu.h>

_extern_c_ void k_cpu_build_hit_points(void *buffers[], void *args);
_extern_c_ void k_cuda_build_hit_points(void *buffers[], void *args);

static struct starpu_codelet cl_build_hit_points = {
  .where = STARPU_CPU/* | STARPU_CUDA*/,
  .cpu_funcs = { k_cpu_build_hit_points, NULL },
  .cuda_funcs = { k_cuda_build_hit_points, NULL },
  .nbuffers = 0,
  //.modes = { STARPU_RW }
};

_extern_c_ void k_build_hit_points() {
  struct starpu_task* task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &cl_build_hit_points;
  //task->handles[0] =
  //task->cl_arg =
  //task->cl_arg_size =

  starpu_task_submit(task);
}
