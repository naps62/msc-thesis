#ifndef _PPM_CUDA_ENGINE_H_
#define _PPM_CUDA_ENGINE_H_

#include "ppm/engine.h"

namespace ppm {

class CUDAEngine : public Engine {
public:
  CUDAEngine(const Config& _config);
  virtual ~CUDAEngine();

protected:
  cudaStream_t stream;

  Seed* seeds;
  EyePath* eye_paths;
  HitPointPosition* hit_points_info;
  HitPointRadiance* hit_points;
  PhotonPath* live_photon_paths;

  BBox* bbox;
  unsigned* hash_grid;
  unsigned* hash_grid_indexes;
  unsigned* hash_grid_lengths;
  float* inv_cell_size;
  float* current_photon_radius2;

  PtrFreeScene* cuda_scene;

  HitPointPosition* host_hit_points_info;
  HitPointRadiance* host_hit_points;
  BBox* host_bbox;
  unsigned* host_hash_grid;
  unsigned* host_hash_grid_indexes;
  unsigned* host_hash_grid_lengths;
  float* host_inv_cell_size;
  float* host_current_photon_radius2;

  Seed* host_seeds;
  EyePath* host_eye_paths;
  PhotonPath* host_photon_paths;

  void output();

  void init_seed_buffer();
  void generate_eye_paths();
  void advance_eye_paths();
  void bbox_compute();
  void rehash();
  void generate_photon_paths();
  void advance_photon_paths();
  void accumulate_flux();
  void update_sample_buffer();
  void splat_to_film();

  void wait_for_all();
  void before();
  void after();
};

namespace kernels { namespace cuda {
  void __global__ init_seeds_impl(
      Seed* const seeds, const unsigned size,
      const unsigned iteration);

  void __global__ generate_eye_paths_impl(
    EyePath* const eye_paths,
    Seed* const seed_buffer,
    const unsigned width,
    const unsigned height,
    const PtrFreeScene* scene);

  void __global__ advance_eye_paths_impl(
    HitPointPosition* const hit_points, //const unsigned hit_points_count
    EyePath*  const eye_paths,            const unsigned eye_paths_count,
    Seed*     const seed_buffer,          //const unsigned seed_buffer_count,
    PtrFreeScene* scene,
    const unsigned max_eye_path_depth);



  void __global__ generate_photon_paths_impl(
    PhotonPath* const photon_paths,
    const unsigned photon_paths_count,
    Seed* const seed_buffer,
    const PtrFreeScene* scene);

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
      const float*              hash_grid_inv_cell_size);

  void __global__ accum_flux_impl(
    const HitPointPosition* const hit_points_info,
    HitPointRadiance* const hit_points,
    const unsigned size,
    const float alpha,
    const unsigned photons_traced,
    const float* current_photon_radius2);
} }

}

#endif // _PPM_CUDA_ENGINE_H_
