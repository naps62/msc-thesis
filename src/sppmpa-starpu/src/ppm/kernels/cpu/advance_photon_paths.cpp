#include "ppm/kernels/codelets.h"
using namespace ppm::kernels::codelets;

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
      PhotonPath* const photon_paths,    const unsigned photon_paths_count,
      Seed* const seed_buffer,        // const unsigned seed_buffer_count,
      const PtrFreeScene* scene,

      HitPointPosition* const hit_points_info,
      HitPointRadiance* const hit_points,
      const BBox& bbox,
      const unsigned CONST_max_photon_depth,
      const float photon_radius2,
      const unsigned hit_points_count,

      const unsigned*           hash_grid,
      const unsigned*           hash_grid_lengths,
      const unsigned*           hash_grid_indexes,
      const float               hash_grid_inv_cell_size,
      const unsigned num_threads) {

  #pragma omp parallel for num_threads(num_threads)
  for(unsigned i = 0; i < photon_paths_count; ++i) {
    PhotonPath& path = photon_paths[i];
    Ray& ray    = path.ray;
    RayHit hit;
    Seed& seed = seed_buffer[i];

    while(!path.done) {
      hit.SetMiss();
      scene->intersect(ray, hit);

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
          f *= surface_color;

          if (!specular_bounce) {
            helpers::add_flux(hash_grid, hash_grid_lengths, hash_grid_indexes, hash_grid_inv_cell_size, bbox, scene, hit_point, shade_N, wo, path.flux, photon_radius2, hit_points_info, hit_points, hit_points_count);
          }

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
  const timeval start_time = my_WallClockTime();

  // cl_args
  const starpu_args args;
  unsigned hit_points_count;
  unsigned iteration;
  starpu_codelet_unpack_args(args_orig, &args, &hit_points_count, &iteration);

  // buffers
  PhotonPath* const photon_paths = reinterpret_cast<PhotonPath* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned photon_paths_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // hit_points_static_info
  HitPointPosition* const hit_points_info = reinterpret_cast<HitPointPosition* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  //const unsigned hit_points_count = STARPU_VECTOR_GET_NX(buffers[1]);
  // hit_points
  HitPointRadiance* const hit_points = reinterpret_cast<HitPointRadiance* const>(STARPU_VECTOR_GET_PTR(buffers[2]));
  //const unsigned hit_points_count = STARPU_VECTOR_GET_NX(buffers[2]);
  // seeds
  Seed* const seed_buffer          = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[3]));
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[3]);

  const BBox* const bbox = (const BBox* const)STARPU_VARIABLE_GET_PTR(buffers[4]);
  const float* const photon_radius2 = (const float* const)STARPU_VARIABLE_GET_PTR(buffers[5]);

  const unsigned*           hash_grid      = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[6]);
  const unsigned*           lengths        = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[7]);
  const unsigned*           indexes        = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[8]);
  const float*              inv_cell_size  = (const float*)    STARPU_VARIABLE_GET_PTR(buffers[9]);

  memset(hit_points, 0, sizeof(HitPointRadiance) * hit_points_count);

  advance_photon_paths_impl(photon_paths, photon_paths_count,
                            seed_buffer,  // seed_buffer_count,
                            args.cpu_scene,
                            hit_points_info,
                            hit_points,
                            *bbox,
                            args.config->max_photon_depth,
                            *photon_radius2,
                            hit_points_count,

                            hash_grid,
                            lengths,
                            indexes,
                            *inv_cell_size,
                            starpu_combined_worker_get_size());

  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, starpu_combined_worker_get_size(), iteration, start_time, end_time, "(7) advance_photon_paths");

}

} } }
