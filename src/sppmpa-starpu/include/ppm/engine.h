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

    delete sample_buffer;
    delete film;
    delete scene;
  }

  virtual void render() {
    this->before();
    start_time = my_WallClockTime();
    info_start();

    this->init_seed_buffer();

    // main loop
    while((!display || display->is_on()) && iteration <= config.max_iters) {
      this->generate_eye_paths();
      this->advance_eye_paths();
      this->bbox_compute();
      this->rehash();
      this->generate_photon_paths();
      this->advance_photon_paths();
      this->accumulate_flux();
      this->update_sample_buffer();
      this->splat_to_film();

      total_photons_traced += config.photons_per_iter;
      iteration++;

      if ((config.max_iters_at_once > 0 && iteration % config.max_iters_at_once == 0)) {
        wait_for_all();
      } else if (config.max_iters_at_once == 0 && display) {
        wait_for_all();
      }

      if (display) {
        set_captions();
        display->request_update(config.min_frame_time);
      }

      if (config.saving_offset > 0 && iteration % config.saving_offset == 0) {
        wait_for_all();
        output(to_string<uint>(iteration, std::dec));
      }
    }


    wait_for_all();
    end_time = my_WallClockTime();
    info_end(start_time, end_time, iteration, total_photons_traced);
    const double us_start = start_time.tv_sec + start_time.tv_usec /  1000000.0;
    const double us_end   = end_time.tv_sec   + end_time.tv_usec / 1000000.0;
    fprintf(stderr, "Total Time:\n%f\n", us_end - us_start);
    this->after();
  }

  void set_captions() {
    const double elapsed_time = WallClockTime() - (start_time.tv_sec + start_time.tv_usec / 1000000.0);
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

  void output(string prefix = string()) {
    film->SaveImpl(config.output_dir + '/' + prefix + config.output_file);
  }

  virtual void init_seed_buffer() = 0;
  virtual void generate_eye_paths() = 0;
  virtual void advance_eye_paths() = 0;
  virtual void bbox_compute() = 0;
  virtual void rehash() = 0;
  virtual void generate_photon_paths() = 0;
  virtual void advance_photon_paths() = 0;
  virtual void accumulate_flux() = 0;
  virtual void update_sample_buffer() = 0;
  virtual void splat_to_film() = 0;

  virtual void wait_for_all() = 0;

  virtual void before() = 0;
  virtual void after() = 0;

protected:
  unsigned iteration;
  unsigned long long total_photons_traced;
  timeval start_time;
  timeval end_time;
  const Config& config;
  PtrFreeScene* scene;
  Display* display;
  Film* film;
  SampleBuffer* sample_buffer;

  unsigned int* hash_grid;
  unsigned int* hash_grid_lengths;
  unsigned int* hash_grid_indexes;
  unsigned long long hash_grid_entry_count;
  float         hash_grid_inv_cell_size;
};

}

#endif // _PPM_ENGINE_H_
