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
  starpu_data_handle_t hash_grid_entry_count_h;
  starpu_data_handle_t hash_grid_inv_cell_size_h;

  starpu_data_filter filter_by_hit_points;
  starpu_data_filter filter_by_photon_paths;

  void init_starpu_handles();

  void render();
  void output();
};

}

#endif // _PPM_STARPU_ENGINE_H_
