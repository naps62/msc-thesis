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
  Engine(const Config& _config)
  :   iteration(1),
      total_photons_traced(0),
      config(_config),
      scene(new PtrFreeScene(config)),
      film(new Film(config.width, config.height)),
      sample_buffer(new SampleBuffer(config.width * config.height * config.spp * config.spp)) {


    film->Reset();

    if (config.use_display) {
      display = new Display(config, *film);
      display->start(true);
    } else {
      display = NULL;
    }
  }

  virtual ~Engine() {
    if (config.use_display) {
      display->join();
    }
  }

  virtual void render() = 0;

  void set_captions() {
    const double elapsed_time = WallClockTime() - start_time;
    const unsigned long long total_photons_M = float(total_photons_traced / 1000000.f);
    const unsigned long long photons_per_sec = total_photons_traced / (elapsed_time * 1000.f);

    stringstream header, footer;
    header << "Hello World!";
    footer << std::setprecision(2) << "[" << total_photons_M << "M Photons]" <<
                                      "[" << photons_per_sec << "K photons/sec]" <<
                                      "[iter: " << iteration << "]" <<
                                      "[" << int(elapsed_time) << "secs]";
    display->set_captions(header, footer);
  }

  void output() {
    film->SaveImpl(config.output_file);
  }

protected:
  unsigned iteration;
  unsigned long long total_photons_traced;
  double start_time;
  const Config& config;
  PtrFreeScene* scene;
  Display* display;
  Film* film;
  SampleBuffer* sample_buffer;

  unsigned int* hash_grid;
  unsigned int* hash_grid_lengths;
  unsigned int* hash_grid_indexes;
  unsigned int  hash_grid_entry_count;
  float         hash_grid_inv_cell_size;
};

}

#endif // _PPM_ENGINE_H_
