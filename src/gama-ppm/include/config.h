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
	string scene_cfg;

	// window
	uint width;
	uint height;
	string title;

	// render
	float alpha;
	uint spp;
	int val_a, val_b;
	bool flag_1, flag_2;

	Config(const char *desc, int _argc, char **_argv);
};

#endif
