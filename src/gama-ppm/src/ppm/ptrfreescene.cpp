/*
 * ptrfreescene.cpp
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#include "ppm/ptrfreescene.h"

//
// constructors
//

// constructor receiving a config struct
//*************************
PtrFreeScene :: PtrFreeScene(const Config& _config)
: config(_config),
  original_scene(new slg::Scene(config.scene_file, -1)) { // TODO what is this -1?
//************************
	ActionList actions;
	actions.add_all();
	recompile(actions);
}

void PtrFreeScene :: recompile(const ActionList& actions) {
	if (actions.has(ACTION_FILM_EDIT) || actions.has(ACTION_CAMERA_EDIT))
		compile_camera();
	if (actions.has(ACTION_GEOMETRY_EDIT))
		compile_geometry();
	// TODO is this one not used?
	//if (actions.has(ACTION_INSTANCE_TRANS_EDIT))
	if (actions.has(ACTION_MATERIALS_EDIT) || actions.has(ACTION_MATERIAL_TYPES_EDIT))
		compile_materials();
	if (actions.has(ACTION_AREA_LIGHTS_EDIT))
		compile_area_lights();
	if (actions.has(ACTION_INFINITE_LIGHT_EDIT))
		compile_infinite_light();
	if (actions.has(ACTION_SUN_LIGHT_EDIT))
		compile_sun_light();
	if (actions.has(ACTION_SKY_LIGHT_EDIT))
		compile_sky_light();
	if (actions.has(ACTION_TEXTURE_MAPS_EDIT))
		compile_texture_maps();
}

/*
 * private methods
 */

void PtrFreeScene :: compile_camera() {

}

void PtrFreeScene :: compile_geometry() {

}

void PtrFreeScene :: compile_materials() {

}

void PtrFreeScene :: compile_area_lights() {

}

void PtrFreeScene :: compile_infinite_light() {

}

void PtrFreeScene :: compile_sun_light() {

}

void PtrFreeScene :: compile_sky_light() {

}

void PtrFreeScene :: compile_texture_maps() {

}
