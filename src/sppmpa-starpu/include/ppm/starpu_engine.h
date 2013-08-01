#ifndef _PPM_STARPU_ENGINE_H_
#define _PPM_STARPU_ENGINE_H_

#include "ppm/engine.h"
#include <starpu.h>

namespace ppm {

class StarpuEngine : public Engine {
public:
  StarpuEngine(const Config& _config);
  ~StarpuEngine();

protected:
  unsigned worker_count;
  BBox bbox;

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

  void render();
  void output();

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

#endif // _PPM_STARPU_ENGINE_H_
