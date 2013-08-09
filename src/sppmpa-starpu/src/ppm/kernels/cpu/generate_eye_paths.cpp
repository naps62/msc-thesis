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

namespace ppm { namespace kernels { namespace cpu {

void generate_eye_paths_impl(
    EyePath* const eye_paths, // const unsigned eye_path_count,
    Seed* const seed_buffer,  // const unsigned seed_buffer_count,
    const unsigned width,
    const unsigned height,
    const PtrFreeScene* scene,
    const unsigned num_threads) {

  #pragma omp parallel for num_threads(num_threads)
  for(unsigned y = 0; y < height; ++y) {
    for(unsigned x = 0; x < width; ++x) {

      unsigned index =  width * y + x;

      EyePath& eye_path = eye_paths[index];

      eye_path = EyePath();
      eye_path.scr_x = x + (/*sx +*/ floatRNG(seed_buffer[index])) /** sample_weight*/ - 0.5f;
      eye_path.scr_y = y + (/*sy +*/ floatRNG(seed_buffer[index])) /** sample_weight*/ - 0.5f;

      float u0 = floatRNG(seed_buffer[index]);
      float u1 = floatRNG(seed_buffer[index]);
      float u2 = floatRNG(seed_buffer[index]);


      eye_path.ray = helpers::generate_ray(eye_path.scr_x, eye_path.scr_y, width, height, u0, u1, u2, scene->camera);

      eye_path.done = false;
      eye_path.sample_index = index;
      ++index;
    }
  }
}


void generate_eye_paths(void* buffers[], void* args_orig) {
  // cl_args
  const starpu_args args;
  starpu_codelet_unpack_args(args_orig, &args);

  // buffers
  // eye_paths
  EyePath* const eye_paths      = reinterpret_cast<EyePath* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  //const unsigned eye_path_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // seeds
  Seed* const seed_buffer          = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[1]);


  generate_eye_paths_impl(eye_paths,   // eye_path_count,
                          seed_buffer, // seed_buffer_count,
                          args.config->width,
                          args.config->height,
                          args.cpu_scene,
                          starpu_combined_worker_get_size());


}

} } }
