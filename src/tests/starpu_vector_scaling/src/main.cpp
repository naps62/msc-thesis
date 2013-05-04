#include <starpu.h>
#include <stdio.h>

#include <starpu_wrapper.hpp>

#include "cpu_kernel.h"

struct params {
  int i;
  float f;
};

void callback_func(void *callback_arg) {
  printf("Callback function (arg %p)\n", callback_arg);
}

int main(int /*argc*/, char ** /*argv*/) {
  float factor = 3.14;
#define NX 100
  float vector[NX];

  starpu::init();

  starpu::codelet cl(STARPU_CPU, cpu_kernel);
  cl
    .nbuffers(1)
    .mode(STARPU_RW);

  starpu_data_handle_t vector_handle;
  starpu_vector_data_register(&vector_handle, 0, (uintptr_t)vector, NX, sizeof(vector[0]));

  starpu::task task(cl);
  task
    .set_sync()
    .handle(vector_handle)
    .arg(&factor)
    .callback(callback_func, (void*) 0x42);

  starpu::submit(task);

  starpu::shutdown();
  return 0;
}
