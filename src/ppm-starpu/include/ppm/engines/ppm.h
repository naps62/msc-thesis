#ifndef _PPM_ENGINES_PPM_H_
#define _PPM_ENGINES_PPM_H_

#include "ppm/engine.h"


namespace ppm {

class PPM : public Engine {
public:
  PPM(const Config& _config);
  ~PPM();

  void render();
  void set_captions();

private:
  Seed* seed_buffer;

  void init_seed_buffer();
  void build_hit_points();
};

}

#endif // _PPM_ENGINES_PPM_H_
