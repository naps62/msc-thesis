#ifndef _PPM_CUDA_ENGINE_H_
#define _PPM_CUDA_ENGINE_H_

#include "ppm/engine.h"

namespace ppm {

class CUDAEngine : public Engine {
public:
  CUDAEngine(const Config& _config);
  ~CUDAEngine();

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

}

#endif // _PPM_CUDA_ENGINE_H_
