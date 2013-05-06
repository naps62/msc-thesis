#include "ppm/engines/ppm.h"

#include "ppm/kernels/build_hit_points.h"

namespace ppm {

PPM :: PPM(const Config& config)
: Engine(config) { }

PPM :: ~PPM() { }

void PPM :: render() {
  film.clear(Spectrum(1.f, 0.f, 0.f));

  k_build_hit_points();

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

}
