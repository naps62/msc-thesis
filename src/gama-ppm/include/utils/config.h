#ifndef _UTILS_CONFIG_H_
#define _UTILS_CONFIG_H_

#include <beast/program_options.hpp>
#include <string>
using std::string;

#include "ppm/types.h"

struct Config : public beast::program_options::options {
	const int argc;
	const char** argv;

	// scene
	string scene_dir;
	string scene_file;
	string scene_cfg;

	// window
	bool no_display;
	bool use_display;  // derived from (!no_display)
	uint width;
	uint height;
	string title;
	uint max_refresh_rate;
	float min_frame_time; // derived from (1 / max_refresh_rate)
	bool vsync;

	// render
	string engine_name;
	string accel_name;
	ppm::AcceleratorType accel_type;
	float alpha;
	uint spp;
	uint total_hit_points; // derived from (width * height * spp^2)

	Config(const char *desc, int _argc, char **_argv);
};

#endif // _UTILS_CONFIG_H_
