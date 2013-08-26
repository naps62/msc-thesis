#include "utils/config.h"

Config :: Config(const char *desc, int _argc, char **_argv)
: beast::program_options::options(desc), argc(_argc), argv((const char**)_argv) {

  // scene
  value("scene",      scene_name, string("kitchen"), "scene name (default = kitchen)");
  //value("scene_dir",  scene_dir,  string("scenes/simple-mat"), "folder where scene files are stored");
  //value("scene_file", scene_file, string("simple-mat.scn"), "to find <scene_dir>/<scene_file>");
  value("output_dir",    output_dir,  string("."), "output image directory");
  value("output_file",   output_file, string("output.png"), "output image file");

  // window
  flag("no_display", no_display, "Supress realtime display?");
  value("width,w",   width,  uint(320),          "window width");
  value("height,h",  height, uint(240),          "window height");
  value("title,t",   title,  string("ppm-starpu"), "window title");
  value("fps",       fps,    uint(60), "maximum FPS");
  flag("vsync",      vsync, "V-Sync. Can cause problems sometimes, so defaults to false");

  // render
  value("alpha,a",   alpha,       float(0.7), "??? still don't know what this is for");
  value("spp",       spp,         uint(1),    "samples per pixel (supersampling)");
  value("accel",     accel_name,  string("QBVH"), "accelerator type [QBVH (default) | BVH | MQBVH)");
  value("engine",    engine, uint(99), "render engine to use [ppm (default) | ... (others to come)]");
  value("photons_iter", photons_first_iter_exp, uint(20),  "to compute amount of photons on first iteration");
  value("max_threads", max_threads, uint(8),  "number of cpu threads");
  value("max_iters",   max_iters,   uint(100), "number of iterations");
  value("max_eye_path_depth", max_eye_path_depth, uint(16), "max eye path depth");
  value("max_photon_depth",   max_photon_depth,   uint(8),  "max photon path depth");
  value("save_offset", saving_offset, uint(0), "save output every X iterations (default 0)");

  // engine
  //value("photons_per_iter", photons_per_iter, unsigned(1024*256), "chunk size for ray and photon buffers (defaults to 1024*256)");
  value("cuda_block_size", cuda_block_size, unsigned(256), "cuda block size (default 512)");

  // starpu
  value("sched", sched_policy, string("peager"), "scheduling policy (pheft (default) | pgreedy)");
  value("partition_size", partition_size, uint(1024), "size of each starpu data partition");
  value("max_iters_at_once", max_iters_at_once, uint(1), "maximum amount of iterations running at once");

  // now parse the arguments
  parse(_argc, _argv);

  // derived values
  //no_display=true;
  use_display = ! no_display;
  min_frame_time = 1.f / fps;
  total_hit_points = width * height * spp * spp;
  photons_per_iter = 1 << photons_first_iter_exp;
  seed_size = std::max(total_hit_points, photons_per_iter);

  accel_type = ppm::ACCEL_QBVH;
  cuda_block_size_sqrt = sqrt(cuda_block_size);

  scene_dir = "scenes/" + scene_name;
  scene_file = scene_name + ".scn";
  scene_file = scene_dir + '/' + scene_file;
}

using std::cout;
using std::stringstream;
boost::mutex config_mutex;

#include <boost/thread.hpp>

void info_start() {
  stringstream ss;
  ss << "<exec>\n";

  boost::lock_guard<boost::mutex> lock(config_mutex);
  cout << ss.str();
}

void task_info(
    const string device,
    const unsigned id,
    const unsigned omp_size,
    const unsigned iteration,
    const double start_time,
    const double end_time,
    const string info) {

  stringstream ss;
  ss << "<task>";
    ss << "<device>" << device << " " << id << "</device>";
    ss << "<omp_size>" << omp_size << "</omp_size>";
    ss << "<name>" << info << "</name>";
    ss << "<iteration>" << iteration << "</iteration>";
    ss << "<start>" << start_time << "</start>";
    ss << "<end>" << end_time << "</end>";
    ss << "<duration>" << end_time - start_time << "</duration>";
  ss << "</task>\n";

  boost::lock_guard<boost::mutex> lock(config_mutex);
  cout << ss.str();
}

void info_end(double start_time, double end_time, unsigned iterations, unsigned long long total_photons) {
  stringstream ss;
    ss << "<iterations>" << iterations << "</iterations>\n";
    ss << "<total_photons>" << total_photons << "</total_photons>\n";
    ss << "<global_start_time>" << start_time << "</global_start_time>\n";
    ss << "<global_end_time>" << end_time << "</global_end_time>\n";
  ss << "</exec>\n";

  boost::lock_guard<boost::mutex> lock(config_mutex);
  cout << ss.str();
}
