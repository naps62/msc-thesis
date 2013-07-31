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
    const unsigned iteration) {

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned i = 0; i < size; ++i) {
    seeds[i] = mwc(i+iteration);
  }
}


void init_seeds(void* buffers[], void* args_orig) {

  // cl_args
  unsigned iteration;
  starpu_codelet_unpack_args(args_orig, &iteration);

  // buffers
  Seed* const seeds = (Seed*) STARPU_VECTOR_GET_PTR(buffers[0]);
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  init_seeds_impl(seeds, size, iteration);
}

} } }
