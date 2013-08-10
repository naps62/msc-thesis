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

void __global__ accum_flux_impl(
    const HitPointPosition* const hit_points_info,
    HitPointRadiance* const hit_points,
    const unsigned size,
    const float alpha,
    const unsigned photons_traced,
    const float* current_photon_radius2) {

  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size)
    return;

  const float radius2 = *current_photon_radius2;

  const HitPointPosition& hpi = hit_points_info[i];
  HitPointRadiance& hp = hit_points[i];

  hp.hits_count++;
  switch (hpi.type) {
    case CONSTANT_COLOR:
      hp.accum_radiance = hpi.throughput;
      break;
    case SURFACE:
      if (hp.accum_photon_count > 0) {
        hp.reflected_flux = hp.accum_reflected_flux;
        hp.accum_photon_count = 0;
        hp.accum_reflected_flux = Spectrum();
      }
      break;
    default:
      assert(false);
  }

  if (hp.hits_count > 0) {
    const double k = 1.0 / (M_PI * radius2 * photons_traced);
    hp.radiance = (hp.accum_radiance + hp.reflected_flux * k);
  }
  hp.hits_count = 0;
  hp.accum_radiance = Spectrum();
  hp.reflected_flux = Spectrum();
}


void accum_flux(void* buffers[], void* args_orig) {

  std::cout << "asd" << std::endl;
  // cl_args
  starpu_args args;
  float alpha;
  unsigned photons_traced;
  starpu_codelet_unpack_args(args_orig, &args, &alpha, &photons_traced);

  // buffers
  const HitPointPosition* const hit_points_info = (const HitPointPosition*)STARPU_VECTOR_GET_PTR(buffers[0]);
        HitPointRadiance*           const hit_points      = (HitPointRadiance*)STARPU_VECTOR_GET_PTR(buffers[1]);
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);

  const float* const photon_radius2 = (const float*)STARPU_VARIABLE_GET_PTR(buffers[2]);

  const unsigned threads_per_block = args.config->cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  printf("accum\n");
  accum_flux_impl
  <<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (hit_points_info,
    hit_points,
    size,
    alpha,
    photons_traced,
    photon_radius2);

  cudaStreamSynchronize(starpu_cuda_get_local_stream());
  CUDA_SAFE(cudaGetLastError());
  printf("accum\n");
}

} } }
