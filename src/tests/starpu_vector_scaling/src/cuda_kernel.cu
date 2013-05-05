#include "cuda_kernel.h"

#include <starpu.h>

static __global__ void cuda_kernel_impl(float *val, unsigned n, float factor) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    val[i] *= factor;
}

extern "C" void cuda_kernel(void *buffers[], void *args) {
  float *factor = (float*) args;

  // length of the vector
  unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);

  // CUDA copy of the vector pointer
  float *val = (float*) STARPU_VECTOR_GET_PTR(buffers[0]);
  printf("asd: %f\n", val);
  unsigned threads_per_block = 64;
  unsigned nblocks = (n + threads_per_block - 1) / threads_per_block;

  cuda_kernel_impl<<<nblocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>(val, n, *factor);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
