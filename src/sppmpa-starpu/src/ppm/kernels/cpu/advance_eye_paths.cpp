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

namespace ppm { namespace kernels { namespace cpu {

void advance_eye_paths_impl(
    HitPointPosition* const hit_points, //const unsigned hit_points_count
    EyePath*  const eye_paths,            const unsigned eye_paths_count,
    Seed*     const seed_buffer,          //const unsigned seed_buffer_count,
    const PtrFreeScene* const scene,
    const unsigned max_eye_path_depth) {

  #pragma omp parallel for num_threads(starpu_combined_worker_get_size())
  for(unsigned i = 0; i < eye_paths_count; ++i) {
    EyePath& eye_path = eye_paths[i];
    Ray&   ray = eye_path.ray; // rays[i];
    RayHit hit;                // = hits[i];

    while(!eye_path.done) {
      hit.SetMiss();
      scene->intersect(ray, hit);

      if (eye_path.depth > max_eye_path_depth) {
        // make it done
        HitPointPosition& hp = hit_points[eye_path.sample_index];
        hp.type = CONSTANT_COLOR;
        hp.scr_x = eye_path.scr_x;
        hp.scr_y = eye_path.scr_y;
        hp.throughput = Spectrum();

        eye_path.done = true;
      } else {
        eye_path.depth++;
      }

      if (hit.Miss()) {
        // add a hit point
        HitPointPosition& hp = hit_points[eye_path.sample_index];
        hp.type = CONSTANT_COLOR;
        hp.scr_x = eye_path.scr_x;
        hp.scr_y = eye_path.scr_y;

        if (scene->infinite_light.exists || scene->sun_light.exists || scene->sky_light.exists) {
          if (scene->infinite_light.exists) {
            // TODO check this
            helpers::infinite_light_le(hp.throughput, eye_path.ray.d, scene->infinite_light, scene->infinite_light_map);
          }
          if (scene->sun_light.exists) {
            // TODO check this
            helpers::sun_light_le(hp.throughput, eye_path.ray.d, scene->sun_light);
          }
          if (scene->sky_light.exists) {
            // TODO check this
            helpers::sky_light_le(hp.throughput, eye_path.ray.d, scene->sky_light);
          }
          hp.throughput *= eye_path.flux;
        } else {
          hp.throughput = Spectrum();
        }
        eye_path.done = true;
      } else {

        // something was hit
        Point hit_point;
        Spectrum surface_color;
        Normal N, shade_N;

        if (helpers::get_hit_point_information(scene, eye_path.ray, hit, hit_point, surface_color, N, shade_N)) {
          continue;
        }

        // get the material
        const unsigned current_triangle_index = hit.index;
        const unsigned current_mesh_index = scene->mesh_ids[current_triangle_index];
        const unsigned material_index = scene->mesh_materials[current_mesh_index];
        const Material& hit_point_mat = scene->materials[material_index];
        unsigned mat_type = hit_point_mat.type;

        if (mat_type == MAT_AREALIGHT) {
          // add a hit point
          HitPointPosition &hp = hit_points[eye_path.sample_index];
          hp.type = CONSTANT_COLOR;
          hp.scr_x = eye_path.scr_x;
          hp.scr_y = eye_path.scr_y;

          Vector md = - eye_path.ray.d;
          helpers::area_light_le(hp.throughput, md, N, hit_point_mat.param.area_light);
          hp.throughput *= eye_path.flux;
          eye_path.done = true;
        } else {
          Vector wo = - eye_path.ray.d;
          float material_pdf;

          Vector wi;
          bool specular_material = true;
          float u0 = floatRNG(seed_buffer[eye_path.sample_index]);
          float u1 = floatRNG(seed_buffer[eye_path.sample_index]);
          float u2 = floatRNG(seed_buffer[eye_path.sample_index]);
          Spectrum f;


          helpers::generic_material_sample_f(hit_point_mat, wo, wi, N, shade_N, u0, u1, u2, material_pdf, f, specular_material);
          f *= surface_color;

          if ((material_pdf <= 0.f) || f.Black()) {
            // add a hit point
            HitPointPosition& hp = hit_points[eye_path.sample_index];
            hp.type = CONSTANT_COLOR;
            hp.scr_x = eye_path.scr_x;
            hp.scr_y = eye_path.scr_y;
            hp.throughput = Spectrum();
          } else if (specular_material || (!hit_point_mat.diffuse)) {
            eye_path.flux *= f / material_pdf;
            eye_path.ray = Ray(hit_point, wi);
          } else {
            // add a hit point
            HitPointPosition& hp = hit_points[eye_path.sample_index];
            hp.type = SURFACE;
            hp.scr_x = eye_path.scr_x;
            hp.scr_y = eye_path.scr_y;
            hp.material_ss   = material_index;
            hp.throughput = eye_path.flux * surface_color;
            hp.position = hit_point;
            hp.wo = - eye_path.ray.d;
            hp.normal = shade_N;
            eye_path.done = true;
          }
        }
      }
    }
  }
}


void advance_eye_paths(void* buffers[], void* args_orig) {

  // cl_args
  const starpu_args args;
  starpu_codelet_unpack_args(args_orig, &args);

  // buffers
  // hit point static info
  HitPointPosition* const hit_points = reinterpret_cast<HitPointPosition* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  //const unsigned hit_points_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // eye paths
  EyePath* const eye_paths = reinterpret_cast<EyePath* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  const unsigned eye_paths_count = STARPU_VECTOR_GET_NX(buffers[1]);
  // seed buffer
  Seed* const seed_buffer = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[2]));
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[2]);



  advance_eye_paths_impl(hit_points, // hit_points_count,
                         eye_paths,         eye_paths_count,
                         seed_buffer, //    seed_buffer_count,
                         args.cpu_scene,
                         args.cpu_config->max_eye_path_depth);
}

} } }
