/*
 * config.cpp
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#include "utils/config.h"

// constructor receiving a config struct
//*************************
Config :: Config(
		const char *desc,	// command line help description
		int _argc,			// number of args
		char **_argv)		// args list, c-style
: beast::program_options::options(desc),
  argc(_argc), argv((const char**)_argv) {
//*************************

	// scene
	value("scene_name", scene_name, string("kitchen"),     "to find scenes/<scene_name>/ directory");
	value("scene_file", scene_file, string("kitchen.scn"), "to find scenes/<scene_name>/<scene_file>");
	value("scene_cfg",  scene_cfg,  string("render.cfg"),  "to find scenes/<scene_name>/<scene_cfg>");

	// window
	flag("no-display", no_display, "Supress realtime display?");
	value("width,w",   width,  uint(640),          "window width");
	value("height,h",  height, uint(480),          "window height");
	value("title,t",   title,  string("gama-ppm"), "window title");

	// render
	value("alpha,a", alpha,      float(0.7), "??? still don't know what this is for");
	value("spp",     spp,        uint(4),    "samples per pixel (supersampling)");
	value("accel",   accel_name, string("QBVH"), "accelerator type [QBVH (default) | BVH | MQBVH)");

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
}
