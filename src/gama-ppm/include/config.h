/*
 * config.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <beast/program_options.hpp>
#include <string>
using std::string;

struct Config : public beast::program_options::options {
	const int argc;
	const char** argv;

	// scene
	string scene_name;
	string scene_file;
	string scene_cfg;

	// window
	bool no_display;
	bool use_display;  // derived from no_display
	uint width;
	uint height;
	string title;

	// render
	float alpha;
	uint spp;

	Config(const char *desc, int _argc, char **_argv);
};

#endif // CONFIG_H_
