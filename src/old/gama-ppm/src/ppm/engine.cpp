#include "ppm/engine.h"
#include "ppm/engines/ppm.h"

// this should be a gama internal
MemorySystem* LowLevelMemAllocator::_memSys = NULL;

namespace ppm {

//
// constructors
//

Engine :: Engine(const Config& _config)
: config(_config), gama(new RuntimeScheduler()), scene(new PtrFreeScene(config)), film(config) {
	// load display if necessary
	if (config.use_display) {
		display = new Display(config, film);
		display->start(true);
	}
}

Engine :: ~Engine() {
	finalize();
}



void Engine :: finalize() {
	// finalize RuntimeSystem
	delete gama;

	// wait for display to close
	if (config.use_display) {
		display->join();
	}
}

//
// public methods
//

// static
Engine* Engine :: instantiate(const Config& config) {
	if (config.engine_name == string("PPM")) {
		return new PPM(config);
	} else {
		throw new string("Invalid engine name" + config.engine_name);
	}
}

void Engine :: build_hit_points(uint iteration) {
	vector<EyePath> todo_eye_paths(config.total_hit_points);
	hit_point_static_info_iteration_copy = vector<HitPointStaticInfo>(config.total_hit_points);

	// TODO SPPM

	const float sample_weight = 1.f / config.spp;

	// for all hitpoints
	uint hit_point_index = 0;
	for(uint y(0); y < config.height; ++y) {
		for(uint x(0); x < config.width; ++x) {
			for(uint sy(0); sy < config.spp; ++sy) {
				for(uint sx(0); sx < config.spp; ++sx) {
					EyePath* eye_path = &todo_eye_paths[hit_point_index];

					eye_path->scr_x = x + (sx /* + TODO RAND */) * sample_weight - 0.5f;
					eye_path->scr_y = y + (sy /* + TODO RAND */) * sample_weight - 0.5f;

					float u0 = 0; /* TODO RAND */
					float u1 = 0; /* TODO RAND */
					float u2 = 0; /* TODO RAND */

					eye_path->ray = scene->generate_ray(eye_path->scr_x, eye_path->scr_y, config.width, config.height, u0, u1, u2);
					eye_path->sample_index = hit_point_index;
					++hit_point_index;
				}
			}
		}
	}
}

}
