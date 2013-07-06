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
  Engine(const Config& _config);
  ~Engine();
  void render();
  void set_captions();

protected:
  const Config& config;
  PtrFreeScene* scene;
  PtrFreeHashGrid hash_grid;
  Display* display;
  BBox bbox;
  Film film;

  // starpu stuff
  starpu_conf spu_conf;

  std::vector<Seed> seeds;
  std::vector<HitPointStaticInfo> hit_points_info;
  std::vector<HitPoint>           hit_points;

  const unsigned chunk_size;

  void init_seed_buffer();
  void build_hit_points();
  void init_radius();
  void advance_photon_paths();

  void eye_paths_to_hit_points(vector<EyePath>& eye_paths);
  void update_bbox();
};

}

#endif // _PPM_ENGINE_H_
