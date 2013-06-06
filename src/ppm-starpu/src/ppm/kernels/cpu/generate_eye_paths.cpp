#include "ppm/kernels/generate_eye_paths.h"

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

void generate_eye_paths(void* buffers[], void* args_orig) {
  // cl_args
  const args_generate_eye_paths* args = (args_generate_eye_paths*) args_orig;
  const Config*       config = static_cast<const Config*>(args->config);
  const PtrFreeScene* scene  = static_cast<const PtrFreeScene*>(args->scene);

  // buffers
  // eye_paths
  EyePath* const eye_paths      = reinterpret_cast<EyePath* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  //const unsigned eye_path_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // seeds
  Seed* const seed_buffer          = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[1]);

  const unsigned width  = config->width;
  const unsigned height = config->height;
  const unsigned spp    = config->spp;

  // generate the eye paths

  unsigned index = 0;
  const float sample_weight = 1.f / spp;

  //printf("%d\n", starpu_combined_worker_get_size());
  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned y = 0; y < height; ++y) {
    for(unsigned x = 0; x < width; ++x) {
      for(unsigned sy = 0; sy < spp; ++sy) {
        for(unsigned sx = 0; sx < spp; ++sx) {

          EyePath& eye_path = eye_paths[index];

          eye_path = EyePath();
          eye_path.scr_x = x + (sx + floatRNG(seed_buffer[index])) * sample_weight - 0.5f;
          eye_path.scr_y = y + (sy + floatRNG(seed_buffer[index])) * sample_weight - 0.5f;

          float u0 = floatRNG(seed_buffer[index]);
          float u1 = floatRNG(seed_buffer[index]);
          float u2 = floatRNG(seed_buffer[index]);

          eye_path.ray = scene->generate_ray(eye_path.scr_x, eye_path.scr_y, width, height, u0, u1, u2);
          eye_path.sample_index = index;
          ++index;
        }
      }
    }
  }

}

} } }
