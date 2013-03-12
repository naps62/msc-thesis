/*
 * ptrfreescene.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef PTRFREESCENE_H_
#define PTRFREESCENE_H_

#include "config.h"

#include <gama.h>

#include "slg/sdl/scene.h"

class PtrFreeScene {

public:
	PtrFreeScene(const Config& config);
	~PtrFreeScene();

private:
	const Config& config;				// reference to global configs
	const slg::Scene* original_scene;	// original scene structure
};

#endif // PTRFREESCENE_H_
