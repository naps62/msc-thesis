/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

#include <cstdlib>
#include <iostream>


#include "config.h"
#include "engine.h"

#include "unistd.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
	// load configurations
	Config config("Options", argc, argv);

	// load render engine
	Engine engine(config);
}
