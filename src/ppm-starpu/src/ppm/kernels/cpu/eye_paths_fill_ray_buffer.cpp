#include "ppm/kernels/kernels.h"
#include "ppm/kernels/helpers.cuh"
using namespace ppm::kernels;

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

void eye_paths_fill_ray_buffer_impl(
    EyePath* const            eye_paths,
    unsigned* const           eye_paths_indexes,
    HitPointStaticInfo* const hit_points_info,
    Ray* const                ray_buffer,
    const unsigned            start,
    const unsigned            end,
    const unsigned            max_eye_path_depth) {

  unsigned current_buffer_index = 0;

  for(unsigned i = start; i < end; ++i) {
    EyePath& eye_path = eye_paths[i];

    if (!eye_path.done) {
      // check if path reached max depth
      if (eye_path.depth < max_eye_path_depth) {
        // make it done
        HitPointStaticInfo& hp = hit_points_info[eye_path.sample_index];
        hp.type = CONSTANT_COLOR;
        hp.scr_x = eye_path.scr_x;
        hp.scr_y = eye_path.scr_y;
        hp.throughput = Spectrum();

        eye_path.done = true;
      } else {
        eye_path.depth++;
        ray_buffer[current_buffer_index]         = eye_path.ray;
        eye_paths_indexes[current_buffer_index] = i;
        current_buffer_index++;
      }
    }
  }
}


void eye_paths_fill_ray_buffer(void* buffers[], void* args_orig) {

  fflush(stdout);
  // cl_args
  const codelets::starpu_eye_paths_fill_ray_buffer_args* args = (const codelets::starpu_eye_paths_fill_ray_buffer_args*) args_orig;

  // buffers
  // eye paths
  EyePath* const eye_paths = reinterpret_cast<EyePath* const>(STARPU_VECTOR_GET_PTR(buffers[0]));

  // eye paths indexes
  unsigned* const eye_paths_indexes = reinterpret_cast<unsigned* const>(STARPU_VECTOR_GET_PTR(buffers[1]));

  // hit point static info
  HitPointStaticInfo* const hit_points_info = reinterpret_cast<HitPointStaticInfo* const>(STARPU_VECTOR_GET_PTR(buffers[2]));

  // ray buffer
  Ray* const ray_buffer = reinterpret_cast<Ray* const>(STARPU_VECTOR_GET_PTR(buffers[3]));

  eye_paths_fill_ray_buffer_impl(eye_paths,
                                 eye_paths_indexes,
                                 hit_points_info,
                                 ray_buffer,
                                 args->start,
                                 args->end,
                                 args->max_eye_path_depth);
}

} } }
