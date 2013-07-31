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

void splat_to_film_impl(
    luxrays::SampleBuffer* const buffer,
    Film* const film,
    const unsigned width,
    const unsigned height) {

  if (buffer->GetSampleCount() > 0) {
    luxrays::SampleFrameBuffer frame_buffer(width, height);
    film->SplatSampleBuffer(&frame_buffer, true, buffer);
    buffer->Reset();
  }
}


void splat_to_film(void* buffers[], void* args_orig) {

  // cl_args
  unsigned width;
  unsigned height;
  starpu_codelet_unpack_args(args_orig, &width, &height);

  // buffers
  luxrays::SampleBuffer** buffer = (luxrays::SampleBuffer**) STARPU_VARIABLE_GET_PTR(buffers[0]);
  Film** film = (Film**) STARPU_VARIABLE_GET_PTR(buffers[1]);

  splat_to_film_impl(*buffer, *film, width, height);
}

} } }
