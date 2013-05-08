#include "ppm/engines/ppm.h"

#include "ppm/kernels/generate_eye_paths.h"
#include "ppm/types.h"
#include "utils/random.h"

namespace ppm {

PPM :: PPM(const Config& config)
: Engine(config) {}

PPM :: ~PPM() { }

void PPM :: render() {
  film.clear(Spectrum(1.f, 0.f, 0.f));
  this->init_seed_buffer();
  this->build_hit_points();

  while(true) {
    set_captions();
    display->request_update(config.min_frame_time);
  }
}

void PPM :: set_captions() {
  stringstream header, footer;
  header << "Hello World!";
  footer << "[Photons " << 0 << "M][Avg. photons/sec " << 0 << "K][Elapsed time " << 0 << "secs]";
  display->set_captions(header, footer);
}

void PPM :: init_seed_buffer() {
  seed_buffer = new Seed[config.total_hit_points];

  // TODO is it worth it to move this to a kernel?
  for(uint i = 0; i < config.total_hit_points; ++i) {
    seed_buffer[i] = mwc(i);
  }
}

void PPM :: build_hit_points() {
  // list of eye paths to generate
  EyePath* eye_paths = new EyePath[config.total_hit_points];

  // eye path generation
  k_generate_eye_paths(eye_paths, config.total_hit_points, sizeof(EyePath),
                       seed_buffer, config.total_hit_points, sizeof(Seed),
                       &config, scene);
}

}
