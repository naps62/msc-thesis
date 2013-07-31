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

void bbox_compute(void* buffers[], void* args_orig) {
  const HitPointPosition* const points = reinterpret_cast<const HitPointPosition* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  BBox* const bbox = reinterpret_cast<BBox* const>(STARPU_VARIABLE_GET_PTR(buffers[1]));
  for(unsigned i = 0; i < size; ++i)
    if (points[i].type == SURFACE) {
      *bbox = Union(*bbox, points[i].position);
    }
}


void bbox_reduce(void* buffers[], void* args_orig) {
  BBox* const bbox_dst = reinterpret_cast<BBox* const>(STARPU_VARIABLE_GET_PTR(buffers[0]));
  BBox* const bbox_src = reinterpret_cast<BBox* const>(STARPU_VARIABLE_GET_PTR(buffers[1]));

  *bbox_dst = Union(*bbox_dst, *bbox_src);
}


void bbox_zero_initialize(void* buffers[], void* args_orig) {
  BBox* const bbox = reinterpret_cast<BBox* const>(STARPU_VARIABLE_GET_PTR(buffers[0]));
  *bbox = BBox();
}

} } }
