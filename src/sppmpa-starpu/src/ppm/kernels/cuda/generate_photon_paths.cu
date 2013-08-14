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

void __global__ generate_photon_paths_impl(
    PhotonPath* const photon_paths,
    const unsigned photon_paths_count,
    Seed* const seed_buffer,
    const PtrFreeScene* scene) {

  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= photon_paths_count)
    return;

  PhotonPath& path = photon_paths[index];
  Ray& ray = path.ray;
  float light_pdf;
  float pdf;
  Spectrum f;

  const float u0 = floatRNG(seed_buffer[index]);
  const float u1 = floatRNG(seed_buffer[index]);
  const float u2 = floatRNG(seed_buffer[index]);
  const float u3 = floatRNG(seed_buffer[index]);
  const float u4 = floatRNG(seed_buffer[index]);

  int light_index;
  ppm::LightType light_type;
  light_type = helpers::sample_all_lights(u0, scene->area_lights_count, scene->infinite_light, scene->sun_light, scene->sky_light, light_pdf, light_index);

  if (light_type == ppm::LIGHT_IL_IS)
    helpers::infinite_light_sample_l(u1, u2, u3, u4, scene->infinite_light, scene->infinite_light_map, scene->bsphere, pdf, ray, path.flux);
  else if (light_type == ppm::LIGHT_SUN)
    helpers::sun_light_sample_l(u1, u2, u3, u4, scene->sun_light, scene->bsphere, pdf, ray, path.flux);
  else if (light_type == ppm::LIGHT_IL_SKY)
    helpers::sky_light_sample_l(u1, u2, u3, u4, scene->sky_light, scene->bsphere, pdf, ray, path.flux);
  else {
    helpers::triangle_light_sample_l(u1, u2, u3, u4, scene->area_lights[light_index], scene->mesh_descs, scene->colors, pdf, ray, path.flux);
  }

  path.flux /= pdf * light_pdf;
  path.depth = 0;
  path.done = 0;
}

void generate_photon_paths(void* buffers[], void* args_orig) {

    const starpu_args args;
    starpu_codelet_unpack_args(args_orig, &args);

    // buffers
    // photon paths
    PhotonPath* const photon_paths = (PhotonPath*)STARPU_VECTOR_GET_PTR(buffers[0]);
    const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);
    // seeds
    Seed* const seed_buffer        = (Seed*)STARPU_VECTOR_GET_PTR(buffers[1]);

  const unsigned threads_per_block = args.config->cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  generate_photon_paths_impl
  <<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (photon_paths,
    size,
    seed_buffer,
    args.gpu_scene);

  cudaStreamSynchronize(starpu_cuda_get_local_stream());
  CUDA_SAFE(cudaGetLastError());

}

} } }
