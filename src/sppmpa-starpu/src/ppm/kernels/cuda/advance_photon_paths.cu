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


void __global__ advance_photon_paths_impl(
      PhotonPath* const photon_paths,    const unsigned photon_paths_count,
      Seed* const seed_buffer,        // const unsigned seed_buffer_count,
      PtrFreeScene* scene,

      HitPointPosition* const hit_points_info,
      HitPointRadiance* const hit_points,
      const BBox* bbox,
      const unsigned CONST_max_photon_depth,
      const float* photon_radius2,
      const unsigned hit_points_count,

      const unsigned*           hash_grid,
      const unsigned*           hash_grid_lengths,
      const unsigned*           hash_grid_indexes,
      const float*              hash_grid_inv_cell_size) {


  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= photon_paths_count)
    return;

    PhotonPath& path = photon_paths[i];
    Ray ray    = path.ray;
    RayHit hit;
    Seed& seed = seed_buffer[i];

    while(!path.done) {

      hit.SetMiss();
      helpers::subIntersect(ray, scene->nodes, scene->prims, hit);
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
            helpers::add_flux(hash_grid, hash_grid_lengths, hash_grid_indexes, *hash_grid_inv_cell_size, *bbox, scene, hit_point, shade_N, wo, path.flux, *photon_radius2, hit_points_info, hit_points, hit_points_count);
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



void advance_photon_paths(void* buffers[], void* args_orig) {
  // cl_args

  const starpu_args args;
  unsigned hit_points_count;
  starpu_codelet_unpack_args(args_orig, &args, &hit_points_count);

  // buffers
  PhotonPath* const photon_paths = (PhotonPath*)STARPU_VECTOR_GET_PTR(buffers[0]);
  const unsigned size = STARPU_VECTOR_GET_NX(buffers[0]);
  // hit_points_static_info
  HitPointPosition* const hit_points_info = (HitPointPosition*)STARPU_VECTOR_GET_PTR(buffers[1]);
  // hit_points
  HitPointRadiance* const hit_points = (HitPointRadiance*)STARPU_VECTOR_GET_PTR(buffers[2]);
  // seeds
  Seed* const seed_buffer          = (Seed*)STARPU_VECTOR_GET_PTR(buffers[3]);

  const BBox* const bbox = (const BBox*)STARPU_VARIABLE_GET_PTR(buffers[4]);
  const float* const photon_radius2 = (const float*)STARPU_VARIABLE_GET_PTR(buffers[5]);

  const unsigned*           hash_grid      = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[6]);
  const unsigned*           lengths        = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[7]);
  const unsigned*           indexes        = (const unsigned*) STARPU_VECTOR_GET_PTR(buffers[8]);
  const float*              inv_cell_size  = (const float*)    STARPU_VARIABLE_GET_PTR(buffers[9]);

  const unsigned threads_per_block = args.config->cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  advance_photon_paths_impl
  <<<n_blocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>
   (photon_paths,
    size,
    seed_buffer,
    args.gpu_scene,
    hit_points_info,
    hit_points,
    bbox,
    args.config->max_photon_depth,
    photon_radius2,
    hit_points_count,

    hash_grid,
    lengths,
    indexes,
    inv_cell_size);

  cudaStreamSynchronize(starpu_cuda_get_local_stream());
  CUDA_SAFE(cudaGetLastError());
}

} } }
