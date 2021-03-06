#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;

#include "ppm/kernels/helpers.cuh"

#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
#include "luxrays/core/pixel/framebuffer.h"
#include "ppm/film.h"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cuda {

void __global__ init_seeds_impl(
    Seed* const seeds, const unsigned size,
    const unsigned iteration) {

  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size)
    return;

  seeds[i] = mwc(i * (1 << iteration));
}


void init_seeds(void* buffers[], void* args_orig) {
  int device_id;
  cudaGetDevice(&device_id);
  const timeval start_time = my_WallClockTime();

  // cl_args
  starpu_args args;
  unsigned iteration;
  starpu_codelet_unpack_args(args_orig, &args, &iteration);

  // buffers
  Seed* const seeds = (Seed*) STARPU_VECTOR_GET_PTR(buffers[0]);
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  // cuda dims
  const unsigned threads_per_block = args.config->cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  init_seeds_impl
  <<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (seeds,
    size,
    iteration);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
  CUDA_SAFE(cudaGetLastError());

  const timeval end_time = my_WallClockTime();
  task_info("GPU", device_id, 0, 0, start_time, end_time, "(1) init_seeds");
}

} } }
