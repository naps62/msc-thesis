#include "ppm/engine.h"
#include "ppm/kernels/codelets.h"
#include "utils/random.h"
#include "ppm/kernels/generate_eye_paths.h"
#include "ppm/types.h"

#include <starpu.h>

namespace ppm {

//
// constructors
//

Engine :: Engine(const Config& _config)
: config(_config), scene(new PtrFreeScene(config)), film(config),
  seeds(config.total_hit_points), hit_points(hit_points) {

  // load display if necessary
  if (config.use_display) {
    display = new Display(config, film);
    display->start(true);
  }

  starpu_init(NULL);
  kernels::codelets::init();
}

Engine :: ~Engine() {
  starpu_shutdown();

  // wait for display to close
  if (config.use_display) {
    display->join();
  }
}

//
// public methods
//
void Engine :: render() {
  film.clear(Spectrum(1.f, 0.f, 0.f));
  this->init_seed_buffer();
  this->build_hit_points();

  while(true) {
    set_captions();
    display->request_update(config.min_frame_time);
  }
}

void Engine :: set_captions() {
  stringstream header, footer;
  header << "Hello World!";
  footer << "[Photons " << 0 << "M][Avg. photons/sec " << 0 << "K][Elapsed time " << 0 << "secs]";
  display->set_captions(header, footer);
}

// static
//Engine* Engine :: instantiate(const Config& config) {
//  if (config.engine_name == string("PPM")) {
//    return new PPM(config);
//  } else {
//    throw new string("Invalid engine name" + config.engine_name);
//  }
//}

//
// private methods
//

void Engine :: init_seed_buffer() {
  // TODO is it worth it to move this to a kernel?
  for(uint i = 0; i < config.total_hit_points; ++i) {
    seeds[i] = mwc(i);
  }
}

void Engine :: build_hit_points() {
  // list of eye paths to generate
  vector<EyePath> eye_paths(config.total_hit_points);

  // eye path generation
  kernels::generate_eye_paths(eye_paths, seeds, &config, scene);
  kernels::eye_paths_to_hit_points(eye_paths, hit_points, seeds, &config, scene);
}

}
