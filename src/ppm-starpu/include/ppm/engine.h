#ifndef _PPM_ENGINE_H_
#define _PPM_ENGINE_H_

#include "utils/common.h"
#include "utils/config.h"
#include "ppm/display.h"
#include "ppm/ptrfreescene.h"
#include "ppm/film.h"
#include "utils/random.h"

#include <vector>

namespace ppm {

class Engine {

public:
  Engine(const Config& _config);
  virtual ~Engine();
  virtual void render() = 0;
  virtual void set_captions() = 0;

  static Engine* instantiate(const Config& _config);

protected:
  const Config& config;
  PtrFreeScene* scene;
  Display* display;
  Film film;

  std::vector<Seed> seeds;
  std::vector<HitPointStaticInfo> hit_point_static_info_iteration_copy;

  void build_hit_points(uint iteration);
};

}

#endif // _PPM_ENGINE_H_
