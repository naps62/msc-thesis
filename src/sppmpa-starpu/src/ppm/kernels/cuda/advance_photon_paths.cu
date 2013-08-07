#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;

#include "ppm/kernels/helpers.cuh"
#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
#include "ppm/kernels/helpers.cuh"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cuda {




void advance_photon_paths(void* buffers[], void* args_orig) {
  // cl_args

  const starpu_args args;
  unsigned hit_points_count;
  starpu_codelet_unpack_args(args_orig, &args, &hit_points_count);

  // buffers
  PhotonPath* const photon_paths = (PhotonPath*)STARPU_VECTOR_GET_PTR(buffers[0]);
  const unsigned photon_paths_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // hit_points_static_info
  HitPointPosition* const hit_points_info = (HitPointPosition*)STARPU_VECTOR_GET_PTR(buffers[1]);
  //const unsigned hit_points_count = STARPU_VECTOR_GET_NX(buffers[1]);
  // hit_points
  HitPointRadiance* const hit_points = (HitPointRadiance*)STARPU_VECTOR_GET_PTR(buffers[2]);
  //const unsigned hit_points_count = STARPU_VECTOR_GET_NX(buffers[2]);
  // seeds
  Seed* const seed_buffer          = (Seed*)STARPU_VECTOR_GET_PTR(buffers[3]);
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[3]);

  const BBox* const bbox = (const BBox*)STARPU_VARIABLE_GET_PTR(buffers[4]);
  const float* const photon_radius2 = (const float*)STARPU_VARIABLE_GET_PTR(buffers[5]);

  const unsigned*           hash_grid      = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[6]);
  const unsigned*           lengths        = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[7]);
  const unsigned*           indexes        = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[8]);
  const float*              inv_cell_size  = (const float*)    STARPU_VARIABLE_GET_PTR(buffers[9]);

  const unsigned threads_per_block = args.config->cuda_block_size;
  const unsigned n_blocks          = photon_paths_count / threads_per_block;

  /*helpers::advance_photon_paths_impl<<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (photon_paths, photon_paths_count,
    seed_buffer,  // seed_buffer_count,
    args.cpu_scene,
    hit_points_info,
    hit_points,
    bbox,
    args.config->max_photon_depth,
    photon_radius2,
    hit_points_count,

    hash_grid,
    lengths,
    indexes,
    inv_cell_size);*/

  cudaStreamSynchronize(starpu_cuda_get_local_stream());

}

} } }
