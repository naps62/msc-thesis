#include "utils/config.h"

Config :: Config(const char *desc, int _argc, char **_argv)
: beast::program_options::options(desc), argc(_argc), argv((const char**)_argv) {

	// scene
	value("scene_dir",  scene_dir,  string("scenes/kitchen"), "folder where scene files are stored");
	value("scene_file", scene_file, string("kitchen.scn"), "to find <scene_dir>/<scene_file>");
	value("scene_cfg",  scene_cfg,  string("render.cfg"),  "to find <scene_dir>/<scene_cfg>");

	// window
	flag("no-display", no_display, "Supress realtime display?");
	value("width,w",   width,  uint(640),          "window width");
	value("height,h",  height, uint(480),          "window height");
	value("title,t",   title,  string("gama-ppm"), "window title");
	value("fps",       max_refresh_rate, uint(60), "maximum FPS");
	flag("vsync",      vsync, "V-Sync. Can cause problems sometimes, so defaults to false");

	// render
	value("alpha,a", alpha,       float(0.7), "??? still don't know what this is for");
	value("spp",     spp,         uint(4),    "samples per pixel (supersampling)");
	value("accel",   accel_name,  string("QBVH"), "accelerator type [QBVH (default) | BVH | MQBVH)");
	value("engine",  engine_name, string("ppm"), "render engine to use [ppm (default) | ... (others to come)]");

	// now parse the arguments
	parse(_argc, _argv);

	// derived values
	use_display = ! no_display;

	if (accel_name == string("BVH"))
		accel_type = ppm::ACCEL_BVH;
	else if (accel_name == string("MQBVH"))
		accel_type = ppm::ACCEL_MQBVH;
	else
		accel_type = ppm::ACCEL_QBVH;

	scene_file = scene_dir + '/' + scene_file;
}
