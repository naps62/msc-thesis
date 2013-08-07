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

  // starpu stuff
  starpu_conf spu_conf;

  starpu_data_handle_t seeds_h;
  starpu_data_handle_t eye_paths_h;
  starpu_data_handle_t hit_points_info_h;
  starpu_data_handle_t hit_points_h;
  starpu_data_handle_t live_photon_paths_h;

  starpu_data_handle_t bbox_h;
  starpu_data_handle_t current_photon_radius2_h;
  starpu_data_handle_t sample_buffer_h;
  starpu_data_handle_t film_h;

  starpu_data_handle_t hash_grid_h;
  starpu_data_handle_t hash_grid_lengths_h;
  starpu_data_handle_t hash_grid_indexes_h;
  starpu_data_handle_t hash_grid_inv_cell_size_h;

  starpu_data_filter filter_by_hit_points;
  starpu_data_filter filter_by_photon_paths;

  unsigned total_spp;

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

  void vector_handle(starpu_data_handle_t* handle, unsigned total, size_t size);
  void variable_handle(starpu_data_handle_t* handle, size_t size);
  void free_handle(starpu_data_handle_t handle);
};

}

#endif // _PPM_STARPU_ENGINE_H_
