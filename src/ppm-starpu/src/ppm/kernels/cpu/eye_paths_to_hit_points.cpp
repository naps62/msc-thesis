#include "ppm/kernels/eye_paths_to_hit_points.h"

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

void intersect_ray_buffer(const PtrFreeScene* scene, RayBuffer& buffer) {
  const unsigned int size = buffer.GetRayCount();

  Ray*    const ray_buffer = buffer.GetRayBuffer();
  RayHit* const hit_buffer = buffer.GetHitBuffer();

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned int i = 0; i < size; ++i) {
    Ray&    ray  = ray_buffer[i];
    RayHit& hit = hit_buffer[i];

    hit.SetMiss();
    scene->intersect(ray, hit);//data_set->Intersect(&ray, &hit);
  }
}

void eye_paths_to_hit_points(void* buffers[], void* args_orig) {
  // cl_args
  const args_eye_paths_to_hit_points* args = (args_eye_paths_to_hit_points*) args_orig;
  const Config*       config = static_cast<const Config*>(args->config);
  const PtrFreeScene* scene  = static_cast<const PtrFreeScene*>(args->scene);

  // buffers
  // eye_paths
  EyePath* const eye_paths      = reinterpret_cast<EyePath* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned eye_path_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // hit_points
  HitPointStaticInfo* const hit_points = reinterpret_cast<HitPointStaticInfo* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned hit_points_count      = STARPU_VECTOR_GET_NX(buffers[0]);

  unsigned todo_eye_paths = eye_path_count;
  unsigned chunk_count = 0;
  const unsigned chunk_size  = 1024 * 256;
  unsigned chunk_done_count = 0;
  RayBuffer ray_buffer(chunk_size);

  while (todo_eye_paths) {

    const unsigned start = chunk_count * chunk_size;
    const unsigned end   = (hit_points_count - start  < chunk_size) ? hit_points_count : start+ chunk_size;

    // 1. fill the ray buffer
    for(unsigned i = start; i < end; ++i) {
      EyePath& eye_path = eye_paths[i];

      if (!eye_path.done) {
        // check if path reached max depth
        if (eye_path.depth > config->max_eye_path_depth) {
          // make it done
          HitPointStaticInfo& hp = hit_points[eye_path.sample_index];
          hp.type  = CONSTANT_COLOR;
          hp.scr_x = eye_path.scr_x;
          hp.scr_y = eye_path.scr_y;
          hp.throughput = Spectrum();

          eye_path.done = true;

        } else {
          // if not, add it to current buffer
          eye_path.depth++;
          ray_buffer.AddRay(eye_path.ray);
        }
      }

      // check if this ray is already done, but not yet splatted
      if (eye_path.done && !eye_path.splat) {
        eye_path.splat = true;
        todo_eye_paths--;
        chunk_done_count++;
        if (chunk_done_count == chunk_size) {
          // move to next chunk
          chunk_count++;
          chunk_done_count = 0;
        }
      }
    }

    // 2. advance ray buffer
    if (ray_buffer.GetRayCount() > 0) {
      intersect_ray_buffer(scene, ray_buffer);

      // advance eye paths

      // reset ray buffer
      ray_buffer.Reset();
    }
  }
}

} } }
