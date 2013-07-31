#include "utils/config.h"

Config :: Config(const char *desc, int _argc, char **_argv)
: beast::program_options::options(desc), argc(_argc), argv((const char**)_argv) {

  // scene
  value("scene_dir",  scene_dir,  string("scenes/kitchen"), "folder where scene files are stored");
  value("scene_file", scene_file, string("kitchen.scn"), "to find <scene_dir>/<scene_file>");
  value("output_dir",    output_dir,  string("."), "output image directory");
  value("output_file",   output_file, string("output.png"), "output image file");

  // window
  flag("no-display", no_display, "Supress realtime display?");
  value("width,w",   width,  uint(320),          "window width");
  value("height,h",  height, uint(240),          "window height");
  value("title,t",   title,  string("ppm-starpu"), "window title");
  value("fps",       fps,    uint(60), "maximum FPS");
  flag("vsync",      vsync, "V-Sync. Can cause problems sometimes, so defaults to false");

  // render
  value("alpha,a",   alpha,       float(0.7), "??? still don't know what this is for");
  value("spp",       spp,         uint(1),    "samples per pixel (supersampling)");
  value("accel",     accel_name,  string("QBVH"), "accelerator type [QBVH (default) | BVH | MQBVH)");
  value("engine",    engine_name, string("PPM"), "render engine to use [ppm (default) | ... (others to come)]");
  value("photons_iter", photons_first_iter_exp, uint(20),  "to compute amount of photons on first iteration");
  value("max_threads", max_threads, uint(8),  "number of cpu threads");
  value("max_iters",   max_iters,   uint(100), "number of iterations");
  value("max_eye_path_depth", max_eye_path_depth, uint(16), "max eye path depth");
  value("max_photon_depth",   max_photon_depth,   uint(8),  "max photon path depth");

  // engine
  value("photons_per_iter", photons_per_iter, unsigned(1024*256), "chunk size for ray and photon buffers (defaults to 1024*256)");

  // starpu
  value("sched", sched_policy, string("pheft"), "scheduling policy (pheft (default) | pgreedy)");
  value("max_iters_at_once", max_iters_at_once, uint(0), "maximum amount of iterations running at once");

  // now parse the arguments
  parse(_argc, _argv);

  // derived values
  use_display = ! no_display;
  min_frame_time = 1.f / fps;
  total_hit_points = width * height * spp * spp;

  if (accel_name == string("BVH"))
    accel_type = ppm::ACCEL_BVH;
  else if (accel_name == string("MQBVH"))
    accel_type = ppm::ACCEL_MQBVH;
  else
    accel_type = ppm::ACCEL_QBVH;

  scene_file = scene_dir + '/' + scene_file;
  output_file = output_dir + '/' + output_file;
}
