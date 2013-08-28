#ifndef _UTILS_CONFIG_H_
#define _UTILS_CONFIG_H_

#include <beast/program_options.hpp>
#include <string>
using std::string;

#include "ppm/types.h"
#include "luxrays/core/utils.h"


struct Config : public beast::program_options::options {
  const int argc;
  const char** argv;

  // scene
  string scene_name;
  string scene_dir;
  string scene_file;
  string output_dir;
  string output_file;

  // window
  bool no_display;
  bool use_display;  // derived from (!no_display)
  uint width;
  uint height;
  string title;
  uint fps;
  float min_frame_time; // derived from (1 / max_refresh_rate)
  bool vsync;

  // render
  uint engine;
  string accel_name;
  float alpha;
  uint spp;
  uint total_hit_points; // derived from (width * height * spp^2)
  uint photons_first_iter_exp;
  uint max_threads;
  uint max_iters;
  ppm::AcceleratorType accel_type;
  uint max_eye_path_depth;
  uint max_photon_depth;
  uint saving_offset;

  // engine
  unsigned photons_per_iter;
  unsigned seed_size;
  unsigned cuda_block_size;
  unsigned cuda_block_size_sqrt;

  // starpu
  string sched_policy;
  uint partition_size;
  uint max_iters_at_once;

  Config(const char *desc, int _argc, char **_argv);
};

void info_start();

void task_info(
    const string device,
    const unsigned id,
    const unsigned omp_size,
    const unsigned iteration,
    const double start_time,
    const double end_time,
    const string info);

void info_end(double start_time, double end_time, unsigned iterations, unsigned long long total_photons);

#endif // _UTILS_CONFIG_H_
