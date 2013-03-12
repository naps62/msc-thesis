/*
 * ptrfreescene.cpp
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#include "ptrfreescene.h"
#include <GL/freeglut.h>

//
// constructors
//

// constructor receiving a config struct
//*************************
PtrFreeScene :: PtrFreeScene(const Config& _config)
: config(_config),
  original_scene(new slg::Scene(config.scene_file, -1)) { // TODO what is this -1?
//************************
}
