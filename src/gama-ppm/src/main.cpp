/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

#include <cstdlib>
#include <iostream>

#include <gama.h>

#include "config.h"
#include "display.h"

MemorySystem* LowLevelMemAllocator::_memSys = NULL;

int main(int argc, char** argv) {
	// load configurations
	Config config("Options", argc, argv);

	RuntimeScheduler* rs = new RuntimeScheduler();

	// create display
	Display display(config);
	display.start();
	display.join();
	delete rs;
}
