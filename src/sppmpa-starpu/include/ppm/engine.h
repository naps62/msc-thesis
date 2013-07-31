#ifndef _PPM_ENGINE_H_
#define _PPM_ENGINE_H_

#include "utils/common.h"
#include "utils/config.h"
#include "ppm/display.h"
#include "ppm/ptrfreescene.h"
#include "ppm/ptrfree_hash_grid.h"
#include "ppm/film.h"
#include "utils/random.h"

#include <vector>
#include <starpu.h>

namespace ppm {

class Engine {

public:
  Engine(const Config& _config, unsigned worker_count);
  ~Engine();
  void render();
  void output();
  void set_captions();

protected:
  unsigned worker_count;
  unsigned iteration;
  float current_photon_radius2;
  unsigned long long total_photons_traced;
  double start_time;
  const Config& config;
  PtrFreeScene* scene;
  PtrFreeHashGrid hash_grid;
  Display* display;
  BBox bbox;
  Film* film;

  // starpu stuff
  starpu_conf spu_conf;

  SampleBuffer* sample_buffer;
  SampleFrameBuffer* frame_buffer;

  std::vector<Seed> seeds;
  std::vector<EyePath> eye_paths;
  std::vector<HitPointPosition> hit_points_info;
  std::vector<HitPointRadiance> hit_points;
  std::vector<PhotonPath> live_photon_paths;

  starpu_data_handle_t seeds_h;
  starpu_data_handle_t eye_paths_h;
  starpu_data_handle_t hit_points_info_h;
  starpu_data_handle_t hit_points_h;
  starpu_data_handle_t live_photon_paths_h;
  starpu_data_handle_t bbox_h;
  starpu_data_handle_t hash_grid_entry_count_h;
  starpu_data_handle_t current_photon_radius2_h;
  starpu_data_handle_t sample_buffer_h;
  starpu_data_handle_t frame_buffer_h;
  starpu_data_handle_t film_h;

  void init_starpu_handles();
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

};

}

#endif // _PPM_ENGINE_H_
