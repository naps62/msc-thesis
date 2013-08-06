#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;

#include "ppm/kernels/helpers.cuh"
#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cuda {


void accum_flux(void* buffers[], void* args_orig) {

  // cl_args
  const starpu_args args;
  unsigned photons_traced;
  starpu_codelet_unpack_args(args_orig, &args, &photons_traced);

  // buffers
  const HitPointPosition* const hit_points_info = (const HitPointPosition*)STARPU_VECTOR_GET_PTR(buffers[0]);
        HitPointRadiance*           const hit_points      = (HitPointRadiance*)STARPU_VECTOR_GET_PTR(buffers[1]);
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  const float* const photon_radius2 = (const float*)STARPU_VARIABLE_GET_PTR(buffers[2]);

  const unsigned threads_per_block = args.config->cuda_block_size;
  const unsigned n_blocks          = size / threads_per_block;

  helpers::accum_flux_impl<<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (hit_points_info,
    hit_points,
    size,
    args.config->alpha,
    photons_traced,
    *photon_radius2);
}

} } }
