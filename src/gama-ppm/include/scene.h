/*
 * scene.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef SCENE_H_
#define SCENE_H_

#include "config.h"

#include <gama.h>

class Scene {

public:
	Scene(const Config& config);
	~Scene();

private:
	const Config& config;
};

#endif // SCENE_H_
