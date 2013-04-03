#ifndef _PPM_ENGINE_H_
#define _PPM_ENGINE_H_

#include "utils/config.h"
#include "ppm/display.h"
#include "ppm/ptrfreescene.h"
#include "ppm/film.h"

#include <gamalib/gamalib.h>

namespace ppm {

class Engine {

public:
	Engine(const Config& _config);
	virtual ~Engine();
	virtual void render() = 0;
	virtual void set_captions() = 0;

	static Engine* get_instance(const Config& config);

protected:
	const Config& config;
	RuntimeScheduler* const gama;
	PtrFreeScene* const scene;
	Display* display;
	Film film;
};

}

#endif // _PPM_ENGINE_H_
