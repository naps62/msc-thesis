/*
 * engine.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef ENGINE_H_
#define ENGINE_H_

#include "config.h"
#include "display.h"

#include <gama.h>

class Engine {

public:
	Engine(const Config& config);
	~Engine();

private:
	const Config& config;			// global configs
	RuntimeScheduler* const gama;	// const pointer, not const data
	Display* display;      			// async display system
};

#endif // ENGINE_H_
