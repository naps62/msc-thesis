#include "ppm/kernels/codelets.h"
#include "ppm/kernels/helpers.cuh"
using namespace ppm::kernels::codelets;
using namespace std;

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


void advance_eye_paths(void* buffers[], void* args_orig) {

  // cl_args
  starpu_args args;
  starpu_codelet_unpack_args(args_orig, &args);

  // buffers
  // hit point static info
  HitPointPosition* const hit_points = (HitPointPosition*)STARPU_VECTOR_GET_PTR(buffers[0]);
  //const unsigned hit_points_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // eye paths
  EyePath* const eye_paths = (EyePath*)STARPU_VECTOR_GET_PTR(buffers[1]);
  const unsigned eye_paths_count = STARPU_VECTOR_GET_NX(buffers[1]);
  // seed buffer
  Seed* const seed_buffer = (Seed*)STARPU_VECTOR_GET_PTR(buffers[2]);
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[2]);


  const unsigned threads_per_block = args.config->cuda_block_size;
  const unsigned n_blocks          = eye_paths_count / threads_per_block;


  helpers::advance_eye_paths_impl<<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (hit_points, // hit_points_count,
    eye_paths,         eye_paths_count,
    seed_buffer, //    seed_buffer_count,
    args.cpu_scene,
    args.config->max_eye_path_depth);
}

} } }
