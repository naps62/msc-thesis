#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;
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

namespace ppm { namespace kernels { namespace cpu {

void init_seeds_impl(
    Seed* const seeds, const unsigned size,
    const unsigned iteration,
    const unsigned num_threads) {

  #pragma omp parallel for num_threads(num_threads)
  for(unsigned i = 0; i < size; ++i) {
    seeds[i] = mwc(i * (1 << iteration));
  }
}


void init_seeds(void* buffers[], void* args_orig) {
  const timeval start_time = my_WallClockTime();

  // cl_args
  starpu_args args;
  unsigned iteration;
  starpu_codelet_unpack_args(args_orig, &args, &iteration);

  // buffers
  Seed* const seeds = (Seed*) STARPU_VECTOR_GET_PTR(buffers[0]);
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  init_seeds_impl(seeds, size, iteration, starpu_combined_worker_get_size());

  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, 0, starpu_combined_worker_get_size(), start_time, end_time, "(1) init_seeds");
}

} } }
