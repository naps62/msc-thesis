#include <starpu.h>
#include <stdio.h>

struct params {
  int i;
  float f;
};

void cpu_func(void * buffers[], void *cl_arg) {
  struct params* params = (struct params*) cl_arg;
  printf("Hello World (params = { %i, %f })\n", params->i, params->f);
}


void callback_func(void *callback_arg) {
  printf("Callback function (arg)\n", callback_arg);
}

int main(int /*argc*/, char ** /*argv*/) {
  starpu_init(NULL);

  struct starpu_codelet cl;
  cl.where = STARPU_CPU;
  cl.cpu_funcs[0] = cpu_func;
  cl.cpu_funcs[1] = NULL;
  cl.nbuffers = 0;


  struct starpu_task* task = starpu_task_create();

  // pointer to the codelet
  task->cl = &cl;

  struct params params = { 1, 2.0f };
  task->cl_arg = &params;
  task->cl_arg_size = sizeof(params);

  task->callback_func = callback_func;
  task->callback_arg  = (void*) 0x42;

  // starpu_task_submit will be a blocking call
  task->synchronous = 1;

  // submit the task to StarPU
  starpu_task_submit(task);

  // terminate StarPU
  starpu_shutdown();

  return 0;
}
