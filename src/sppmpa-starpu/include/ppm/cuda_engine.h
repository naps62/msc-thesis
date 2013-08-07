#ifndef _PPM_CUDA_ENGINE_H_
#define _PPM_CUDA_ENGINE_H_

#include "ppm/engine.h"

namespace ppm {

class CUDAEngine : public Engine {
public:
  CUDAEngine(const Config& _config);
  ~CUDAEngine();

protected:
  BBox bbox;
  float current_photon_radius2;

  Seed seeds;
  EyePath* eye_paths;
  HitPointPosition* hit_points_info;
  HitPointRadiance* hit_points;
  PhotonPath* live_photon_paths;

  unsigned* hash_grid;
  unsigned* hash_grid_indexes;
  unsigned* hash_grid_lengths;
  float inv_cell_size;

  PtrFreeScene* cuda_scene;

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
