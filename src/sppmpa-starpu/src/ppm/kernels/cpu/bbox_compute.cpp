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

void bbox_compute_impl(
    const HitPointPosition* const points, const unsigned size,
    BBox& bbox,
    float& photon_radius2,
    const float iteration,
    const float total_spp,
    const float alpha) {

  bbox = BBox();

  for(unsigned i = 0; i < size; ++i) {
    if (points[i].type == SURFACE) {
      bbox = Union(bbox, points[i].position);
    }
  }

  const Vector ssize = bbox.pMax - bbox.pMin;
  const float photon_radius = ((ssize.x + ssize.y + ssize.z) / 3.f) / (total_spp / 2.f) * 2.f;
  photon_radius2 = photon_radius * photon_radius;

  float g = 1;
  for(uint k = 1; k < iteration; ++k)
    g *= (k + alpha) / k;

  g /= iteration;
  photon_radius2 *= g;
  bbox.Expand(sqrt(photon_radius2));
}


void bbox_compute(void* buffers[], void* args_orig) {
  const double start_time = WallClockTime();

  unsigned iteration;
  unsigned total_spp;
  float alpha;
  starpu_codelet_unpack_args(args_orig, &iteration, &total_spp, &alpha);

  const HitPointPosition* const points = reinterpret_cast<const HitPointPosition* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  BBox* const bbox = reinterpret_cast<BBox* const>(STARPU_VARIABLE_GET_PTR(buffers[1]));
  float* const photon_radius2 = (float*) STARPU_VARIABLE_GET_PTR(buffers[2]);

  bbox_compute_impl(points, size,
                    *bbox,
                    *photon_radius2,
                    iteration,
                    total_spp,
                    alpha);

  const double end_time = WallClockTime();
  task_info("CPU", 0, 1, iteration, start_time, end_time, "(4) bbox_compute");

}

} } }
