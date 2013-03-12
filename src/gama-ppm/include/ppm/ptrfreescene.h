/*
 * ptrfreescene.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_PTRFREESCENE_H_
#define _PPM_PTRFREESCENE_H_

#include "utils/config.h"
#include "slg/sdl/scene.h"

#include <gama.h>


class PtrFreeScene {

public:
	PtrFreeScene(const Config& config);
	~PtrFreeScene();

	void recompile(const ActionList& actions);

private:
	const Config& config;				// reference to global configs
	const slg::Scene* original_scene;	// original scene structure

	void compile_camera();
	void compile_geometry();
	void compile_materials();
	void compile_area_lights();
	void compile_infinite_light();
	void compile_sun_light();
	void compile_sky_light();
	void compile_texture_maps();
};

#endif // _PPM_PTRFREESCENE_H_
