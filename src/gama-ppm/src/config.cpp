#include "config.h"

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
	value("scene_name", scene_name, string("kitchen"), "used to find scenes/<scene_name>/ directory");
	value("scene_cfg",  scene_cfg,  string("render.cfg"), "cfg file inside scene directory");

	// window
	value("width,w",  width,  uint(640), "window width");
	value("height,h", height, uint(480), "window height");
	value("title,t",  title,  string("gama-ppm"), "window title");

	// render
	value("alpha,a", alpha, float(0.7), "??? still don't know what this is for");
	value("spp",     spp,   uint(4),    "samples per pixel (supersampling)");

	// now parse the arguments
	parse(_argc, _argv);
}
