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

void __global__ generate_eye_paths_impl(
    EyePath* const eye_paths,
    Seed* const seed_buffer,
    const unsigned width,
    const unsigned height,
    const PtrFreeScene* scene) {

  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;


  const unsigned index = y * width + x;

  EyePath& eye_path = eye_paths[index];

  eye_path = EyePath();
  eye_path.scr_x = x + (floatRNG(seed_buffer[index])) - 0.5f;
  eye_path.scr_y = y + (floatRNG(seed_buffer[index])) - 0.5f;

  float u0 = floatRNG(seed_buffer[index]);
  float u1 = floatRNG(seed_buffer[index]);
  float u2 = floatRNG(seed_buffer[index]);

  eye_path.ray = helpers::generate_ray(eye_path.scr_x, eye_path.scr_y, width, height, u0, u1, u2, scene->camera);

  eye_path.done = false;
  eye_path.sample_index = index;
}


void generate_eye_paths(void* buffers[], void* args_orig) {
  // cl_args
  const starpu_args args;
  starpu_codelet_unpack_args(args_orig, &args);

  // buffers
  // eye_paths
  EyePath* const eye_paths = (EyePath*)STARPU_VECTOR_GET_PTR(buffers[0]);
  //const unsigned eye_path_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // seeds
  Seed* const seed_buffer  = (Seed*)STARPU_VECTOR_GET_PTR(buffers[1]);
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[1]);

  const unsigned width = args.config->width;
  const unsigned height = args.config->height;
  const unsigned block_side = args.config->cuda_block_size_sqrt;
  const dim3 threads_per_block = dim3(block_side,                block_side);
  const dim3 n_blocks          = dim3(std::ceil(width/(float)threads_per_block.x), std::ceil(height/(float)threads_per_block.y));

  int device_id;
  cudaGetDevice(&device_id);

  generate_eye_paths_impl<<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (eye_paths,   // eye_path_count,
    seed_buffer, // seed_buffer_count,
    width,
    height,
    args.gpu_scene[device_id]);

  cudaStreamSynchronize(starpu_cuda_get_local_stream());
  CUDA_SAFE(cudaGetLastError());
}

} } }
