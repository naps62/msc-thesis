#include <starpu.h>
#include <stdio.h>

#include <starpu_wrapper.hpp>

struct params {
  int i;
  float f;
};

void cpu_func(void * /*buffers*/[], void *cl_arg) {
  struct params* params = (struct params*) cl_arg;
  printf("Hello World (params = { %i, %f })\n", params->i, params->f);
}


void callback_func(void *callback_arg) {
  printf("Callback function (arg %p)\n", callback_arg);
}

int main(int /*argc*/, char ** /*argv*/) {
  starpu::init();

  starpu::codelet cl(STARPU_CPU, cpu_func);


  starpu::task task(cl);
  task.set_sync();
  //struct starpu_task* task = starpu_task_create();

  // pointer to the codelet
  task.ptr()->cl = cl.ptr();

  struct params params = { 1, 2.0f };
  task.ptr()->cl_arg = &params;
  task.ptr()->cl_arg_size = sizeof(params);

  task.ptr()->callback_func = callback_func;
  task.ptr()->callback_arg  = (void*) 0x42;

  // starpu_task_submit will be a blocking call
  //task->synchronous = 1;

  // submit the task to StarPU
  starpu_task_submit(task.ptr());

  // terminate StarPU
  starpu::shutdown();

  return 0;
}
