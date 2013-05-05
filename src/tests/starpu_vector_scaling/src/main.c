#include <starpu.h>
#include <stdio.h>

#include <starpu.h>
#include "cpu_kernel.h"
#include "cuda_kernel.h"

struct params {
  int i;
  float f;
};

static struct starpu_codelet cl = {
  .where = STARPU_CPU | STARPU_CUDA,
  .cpu_funcs = { cpu_kernel, NULL },
  .cuda_funcs = { cuda_kernel, NULL },
  .nbuffers = 1,
  .modes = { STARPU_RW }
};

void callback_func(void *callback_arg) {
  printf("Callback function (arg %p)\n", callback_arg);
}

int main(int argc, char ** argv) {
  float factor = 3.14;
#define NX 5
  float values[NX];

  int i;
  for(i = 0; i < NX; ++i)
    values[i] = i;

  starpu_init(NULL);

  starpu_data_handle_t vector_handle;
  starpu_vector_data_register(&vector_handle, 0, (uintptr_t)&values[0], NX, sizeof(float));

  struct starpu_task *task = starpu_task_create();
  task->synchronous = 1;

  task->cl = &cl;
  task->handles[0] = vector_handle;
  task->cl_arg = &factor;
  task->cl_arg_size = sizeof(float);

  starpu_task_submit(task);

  starpu_data_unregister(vector_handle);
  starpu_shutdown();

  for(i = 0; i < NX; ++i)
    printf("%f\n", values[i]);

  return 0;
}
