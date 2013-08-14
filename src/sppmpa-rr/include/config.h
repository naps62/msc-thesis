/*
 * config.h
 *
 *  Created on: Nov 9, 2012
 *      Author: rr
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#define DISABLE_TIME_BREAKDOWN

#define REBUILD_HASH

#define MAX_ITERATIONS 20000

//#define USE_PPM
/**
 * PPM
 * Single device.
 * Dependant iterations, single build hitpoints, reduce radius and reflected flux.
 * Radius per iteration, dependant and per hit point.
 * Keep local statistics.
 */

//#define USE_SPPM
/**
 * SPPM
 * Single device.
 * Dependant iterations, in each iterations build hitpoints, reduce radius and reflected flux.
 * Radius per iteration, dependant and per hit point.
 * Keep local statistics.
 */

#define USE_SPPMPA
/**
 * SPPM:PA
 * Each device builds hitpoints and hash.
 * Iterations independent, radius not reduced -> precalculated.
 * Radius per iteration, not per hitpoint.
 * 1 inital SPP.
 * Paper PPM:PA approach reversed.
 */

//#define USE_PPMPA
/**
 * PPM:PA
 * Single hit points, each device mirrors hpts and builds hash grid.
 * Iterations independent, radius not reduced.
 * Oversampling.
 * Multi-resolution grid targeted.
 */

#ifdef DISABLE_TIME_BREAKDOWN
#define USE_GLUT
#endif

//#define RENDER_FAST_PHOTON
#define RENDER_TINY
//#define RENDER_MEDIUM
//#define RENDER_BIG
//#define RENDER_HUGE


#define CPU
//#define GPU0
//#define GPU2


#define SM 15
#define FACTOR 256
#define BLOCKSIZE 512

#define MAX_EYE_PATH_DEPTH 16
#define MAX_PHOTON_PATH_DEPTH 8

#define QBVH_STACK_SIZE 24

//#define WARP_RR


#ifndef _UTILS_CONFIG_H_
#define _UTILS_CONFIG_H_

#include <beast/program_options.hpp>
#include <string>
using std::string;

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
  string engine_name;
  string accel_name;
  float alpha;
  uint spp;
  uint total_hit_points; // derived from (width * height * spp^2)
  uint photons_first_iter_exp;
  uint max_threads;
  uint max_iters;

  unsigned engine_chunk_size;

  Config(const char *desc, int _argc, char **_argv)
  : beast::program_options::options(desc), argc(_argc), argv((const char**)_argv) {

    // scene
    value("scene",      scene_name, string("simple-mat"), "scene name (default = simple-mat)");
    //value("scene_dir",  scene_dir,   string("scenes/simple-mat"), "folder where scene files are stored");
    //value("scene_file", scene_file,  string("simple-mat.scn"), "to find <scene_dir>/<scene_file>");
    value("output_dir",    output_dir,  string("."), "output image directory");
    value("output_file",   output_file, string("output.png"), "output image file");

    // window
    flag("no_display", no_display, "Supress realtime display?");
    value("width",   width,  uint(320),          "window width");
    value("height",  height, uint(240),          "window height");
    value("title,t",   title,  string("gama-ppm"), "window title");
    value("fps",       fps,    uint(60), "maximum FPS");
    flag("vsync",      vsync, "V-Sync. Can cause problems sometimes, so defaults to false");

    // render
    value("alpha,a",   alpha,       float(0.7), "??? still don't know what this is for");
    value("spp",       spp,         uint(1),    "samples per pixel (supersampling)");
    value("accel",     accel_name,  string("QBVH"), "accelerator type [QBVH (default) | BVH | MQBVH)");
    value("engine",    engine_name, string("ppm"), "render engine to use [ppm (default) | ... (others to come)]");
    value("photons_iter", photons_first_iter_exp, uint(20),  "to compute amount of photons on first iteration");
    value("max_threads", max_threads, uint(1),  "number of cpu threads");
    value("max_iters",   max_iters,   uint(10), "number of iterations");

    value("chunk_size", engine_chunk_size, unsigned(1024*256), "chunk size for ray and photon buffers (defaults to 1024*256)");

    // now parse the arguments
    parse(_argc, _argv);

    // derived values
    use_display = ! no_display;
    min_frame_time = 1.f / fps;
    total_hit_points = width * height * spp * spp;

    scene_dir = "scenes/" + scene_name;
    scene_file = scene_name + ".scn";
    scene_file  = scene_dir  + '/' + scene_file;
    output_file = output_dir + '/' + output_file;
  }

};

#endif // _UTILS_CONFIG_H_

#endif /* CONFIG_H_ */
