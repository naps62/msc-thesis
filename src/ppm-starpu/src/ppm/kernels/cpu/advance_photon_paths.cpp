#include "ppm/kernels/advance_photon_paths.h"

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

namespace ppm { namespace kernels { namespace cpu {

  void advance_photon_paths_impl(
      Ray* const rays,                   const unsigned rays_count,
      RayHit* const hits,             // const unsigned hits_count,
      PhotonPath* const photon_paths,    const unsigned photon_paths_count,
      Seed* const seed_buffer,        // const unsigned seed_buffer_count,
      const PtrFreeScene* scene,
      const unsigned CONST_max_photon_depth) {

#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
    for(unsigned i = 0; i < rays_count; ++i) {
      Ray& ray    = rays[i];
      RayHit& hit = hits[i];
      PhotonPath& path = photon_paths[i];
      Seed& seed = seed_buffer[i];

      if (path.done)
        continue;

      if (hit.Miss()) {
        path.done = true;
      } else {
        Point hit_point;
        Spectrum surface_color;
        Normal N;
        Normal shade_N;

        if (helpers::get_hit_point_information(scene, ray, hit, hit_point, surface_color, N, shade_N))
          continue;

        const unsigned current_triangle_index = hit.index;
        const unsigned current_mesh_index     = scene->mesh_ids[current_triangle_index];
        const unsigned material_index         = scene->mesh_materials[current_mesh_index];
        const Material& hit_point_mat = scene->materials[material_index];
        unsigned mat_type = hit_point_mat.type;

        if (mat_type == MAT_AREALIGHT) {
          path.done = true;
        } else {
          float f_pdf;
          Vector wi;
          Vector wo = -ray.d;
          Spectrum f;
          bool specular_bounce = true;

          const float u0 = floatRNG(seed);
          const float u1 = floatRNG(seed);
          const float u2 = floatRNG(seed);

          helpers::generic_material_sample_f(hit_point_mat, wo, wi, N, shade_N, u0, u1, u2, f_pdf, f, specular_bounce);

          if (specular_bounce) {
            // add flux

            if (path.depth < CONST_max_photon_depth) {
              if (f_pdf <= 0.f || f.Black()) {
                path.done = true;
              } else {
                path.depth++;
                path.flux *= f / f_pdf;

                // russian roulette
                const float p = 0.75;
                if (path.depth < 3) {
                  ray = Ray(hit_point, wi);
                } else if (floatRNG(seed) < p) {
                  path.flux /= p;
                  ray = Ray(hit_point, wi);
                } else {
                  path.done = true;
                }
              }
            } else {
              path.done = true;
            }
          }
        }
      }
    }
  }



void advance_photon_paths(void* buffers[], void* args_orig) {
  // cl_args
  const args_advance_photon_paths* args = (args_advance_photon_paths*) args_orig;
  const Config*       config = static_cast<const Config*>(args->config);
  const PtrFreeScene* scene  = static_cast<const PtrFreeScene*>(args->scene);

  // buffers
  // rays
  Ray* const rays           = reinterpret_cast<Ray* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned rays_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // hits
  RayHit* const hits = reinterpret_cast<RayHit* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  // const unsigned hits_count = STARPU_VECTOR_GET_NX(buffers[1]);
  // photon_paths
  PhotonPath* const photon_paths = reinterpret_cast<PhotonPath* const>(STARPU_VECTOR_GET_PTR(buffers[2]));
  const unsigned photon_paths_count = STARPU_VECTOR_GET_NX(buffers[2]);
  // seeds
  Seed* const seed_buffer          = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[3]));
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[3]);


  advance_photon_paths_impl(rays,         rays_count,
                            hits,         // hits_count
                            photon_paths, photon_paths_count,
                            seed_buffer,  // seed_buffer_count,
                            scene,
                            config->max_photon_depth);


}

} } }
