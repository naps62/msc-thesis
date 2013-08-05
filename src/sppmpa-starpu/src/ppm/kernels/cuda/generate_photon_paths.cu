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

void generate_photon_paths(void* buffers[], void* args_orig) {

    const starpu_args args;
    starpu_codelet_unpack_args(args_orig, &args);

    // buffers
    // photon paths
    PhotonPath* const photon_paths = (PhotonPath*)STARPU_VECTOR_GET_PTR(buffers[0]);
    const unsigned photon_paths_count = STARPU_VECTOR_GET_NX(buffers[0]);
    // seeds
    Seed* const seed_buffer        = (Seed*)STARPU_VECTOR_GET_PTR(buffers[1]);

  const unsigned threads_per_block = args.config->cuda_block_size;
  const unsigned n_blocks          = photon_paths_count / threads_per_block;

  helpers::generate_photon_paths_impl<<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (photon_paths,
    photon_paths_count,
    seed_buffer,
    args.gpu_scene);


}

} } }
