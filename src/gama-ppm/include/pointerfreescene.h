/*
 * scene.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef POINTERFREESCENE_H_
#define POINTERFREESCENE_H_

#include "config.h"

#include <gama.h>

#include "slg/sdl/scene.h"

class PointerFreeScene {

public:
	PointerFreeScene(const Config& config);
	~PointerFreeScene();

private:
	const Config& config;

	slg::Scene* original_scene;	// original scene structure
};

#endif // POINTERFREESCENE_H_
