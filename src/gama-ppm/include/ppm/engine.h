#ifndef _PPM_ENGINE_H_
#define _PPM_ENGINE_H_

#include "utils/config.h"
#include "ppm/display.h"
#include "ppm/ptrfreescene.h"
#include "ppm/film.h"

#include <gamalib/gamalib.h>
#include "engines/ppm.h"

namespace ppm {

class Engine {

public:
	Engine(const Config& _config);
	virtual ~Engine() = 0;
	virtual void render() = 0;
	virtual void set_captions() = 0;

	static Engine* instantiate(const Config& _config);
	// static Engine* get_instance(const Config& config);

protected:
	const Config& config;
	RuntimeScheduler* const gama;
	PtrFreeScene* const scene;
	Display* display;
	Film film;

	vector<HitPointStaticInfo> hit_point_static_info_iteration_copy;

	void build_hit_points(uint iteration);
	virtual void finalize();
};

}

#endif // _PPM_ENGINE_H_
