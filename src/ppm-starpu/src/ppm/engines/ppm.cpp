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
  // TODO is it worth it to move this to a kernel?
  for(uint i = 0; i < config.total_hit_points; ++i) {
    seeds[i] = mwc(i);
  }
}

void PPM :: build_hit_points() {
  // list of eye paths to generate
  vector<EyePath> eye_paths(config.total_hit_points);

  // eye path generation
  kernels::generate_eye_paths(eye_paths, seeds, &config, scene);
  kernels::eye_paths_to_hit_points(eye_paths, hit_points, seeds, &config, scene);
}

}
