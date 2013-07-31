#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;
#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
#include "utils/common.h"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cpu {

void update_sample_buffer_impl(
    const HitPointRadiance* const hit_points,
    const unsigned size,
    const unsigned width,
    SampleBuffer* const buffer) {

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned i = 0; i < size; ++i) {
    const HitPointRadiance& hp = hit_points[i];

    const float scr_x = i % width;
    const float scr_y = i / width;

    buffer->SplatSample(scr_x, scr_y, hp.radiance);
  }
}


void update_sample_buffer(void* buffers[], void* args_orig) {

  // cl_args
  unsigned width;
  starpu_codelet_unpack_args(args_orig, &width);

  // buffers
  const HitPointRadiance* const hit_points = (const HitPointRadiance* const)(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  SampleBuffer** buffer = (SampleBuffer**) STARPU_VARIABLE_GET_PTR(buffers[1]);

  update_sample_buffer_impl(hit_points, size, width, *buffer);
}

} } }
