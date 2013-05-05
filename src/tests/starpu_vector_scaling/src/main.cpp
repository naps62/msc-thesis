#include <starpu.h>
#include <stdio.h>

#include <starpu_wrapper.hpp>

#include "cpu_kernel.h"
#include "cuda_kernel.h"

struct params {
  int i;
  float f;
};

void callback_func(void *callback_arg) {
  printf("Callback function (arg %p)\n", callback_arg);
}

int main(int /*argc*/, char ** /*argv*/) {
  float factor = 3.14;
#define NX 5
  starpu::vector<float> values(NX);

  for(int i = 0; i < NX; ++i)
    values[i] = i;

  starpu::init();

  starpu::codelet cl(STARPU_CUDA, cpu_kernel, cuda_kernel);
  cl.buffer(STARPU_RW);

  values.register_data();
  //starpu_data_handle_t vector_handle;
  //starpu_vector_data_register(&vector_handle, 0, (uintptr_t)vector, NX, sizeof(vector[0]));

  starpu::task task(cl);
  task
    .set_sync()
    .handle(values)
    .arg(&factor)
    .callback(callback_func, (void*) 0x42);

  starpu::submit(task);
  starpu_task_wait_for_all();

  values.acquire();

  for(int i = 0; i < NX; ++i)
    printf("%f\n", values[i]);

  starpu::shutdown();
  return 0;
}
