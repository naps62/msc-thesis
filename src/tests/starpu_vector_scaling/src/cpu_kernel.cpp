#include "cpu_kernel.h"

#include <starpu.h>

void cpu_kernel(void * buffers[], void *cl_arg) {
  unsigned int i;
  float *factor = (float*) cl_arg;

  // length of the vector
  unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);

  // CPU copy of the vector pointer
  float *val = (float*) STARPU_VECTOR_GET_PTR(buffers[0]);

  for(i = 0; i < n; ++i) {
    val[i] *= *factor;
  }
}
