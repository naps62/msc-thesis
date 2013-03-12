/*
 * display.cpp
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#include "engine.h"
#include <GL/freeglut.h>

// this should be a gama internal
MemorySystem* LowLevelMemAllocator::_memSys = NULL;

//
// constructors
//

// constructor receiving a config struct
//*************************
Engine :: Engine(const Config& _config)
: config(_config),                       // store reference to global configs
  gama(new RuntimeScheduler()),          // pre-load GAMA runtime scheduler
  scene(new PointerFreeScene(config)) {  // pre-load input scene
//************************
	// load display if necessary
	if (config.use_display) {
		display = new Display(config);
		display->start();
	}
}

// destructor
//*************************
Engine :: ~Engine() {
//*************************
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
