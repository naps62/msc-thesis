/*
 * scene.cpp
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#include "pointerfreescene.h"
#include <GL/freeglut.h>

//
// constructors
//

// constructor receiving a config struct
//*************************
PointerFreeScene :: PointerFreeScene(const Config& _config)
: config(_config) {
//************************

	scene = new slg::Scene(config.scene_file, -1); // TODO what is this -1?

}
