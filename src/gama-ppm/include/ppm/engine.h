/*
 * engine.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_ENGINE_H_
#define _PPM_ENGINE_H_

#include "utils/config.h"
#include "ppm/display.h"
#include "ppm/ptrfreescene.h"

#include <gama.h>

namespace ppm {

class Engine {

public:
	Engine(const Config& _config);
	~Engine();

private:
	const Config& config;         // reference to global configs
	RuntimeScheduler* const gama; // const pointer, not const data
	PtrFreeScene* const scene;    // the input scene
	Display* display;             // async display system
};

}

#endif // _PPM_ENGINE_H_
