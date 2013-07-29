#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;

#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cpu {


void bbox_reduce(void* buffers[], void* args_orig) {
  BBox* const bbox = reinterpret_cast<BBox* const>(STARPU_VARIABLE_GET_PTR(buffers[0]));
  const HitPointPosition* const point = reinterpret_cast<const HitPointPosition* const>(STARPU_VARIABLE_GET_PTR(buffers[1]));

  if (point->type == SURFACE)
    *bbox = Union(*bbox, point->position);
}


void bbox_zero_initialize(void* buffers[], void* args_orig) {
  BBox* const bbox = reinterpret_cast<BBox* const>(STARPU_VARIABLE_GET_PTR(buffers[0]));
  *bbox = BBox();
}

} } }
