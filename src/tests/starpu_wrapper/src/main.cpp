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

  struct params params = { 1, 2.0f };

  starpu::task task(cl);
  task
    .set_sync()
    .arg(&params)
    .callback(callback_func, (void*) 0x42);

  starpu::submit(task);

  // terminate StarPU
  starpu::shutdown();

  return 0;
}
