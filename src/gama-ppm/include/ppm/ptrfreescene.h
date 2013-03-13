/*
 * ptrfreescene.h
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_PTRFREESCENE_H_
#define _PPM_PTRFREESCENE_H_

#include "utils/config.h"
#include "ppm/types.h"
#include "slg/sdl/scene.h"
#include "luxrays/core/dataset.h"
#include "utils/action_list.h"
#include "gama_ext/vector.h"


#include <gama.h>

namespace ppm {

class PtrFreeScene {

public:
	PtrFreeScene(const Config& config);
	~PtrFreeScene();

	void recompile(const ActionList& actions);

private:
	const Config& config;				// reference to global configs
	slg::Scene* original_scene;	// original scene structure
	luxrays::DataSet* data_set;   // original data_set structure
	Camera camera;                      // compiled camera

	gama::vector<Point>       vertices;
	gama::vector<Normal>      normals;
	gama::vector<Spectrum>    colors;
//	gama::vector<UV>          uvs;
//	gama::vector<Triangle>    triangles;
//	gama::vector<PtrFreeMesh> meshDescs;

	void compile_camera();
	void compile_geometry();
	void compile_materials();
	void compile_area_lights();
	void compile_infinite_light();
	void compile_sun_light();
	void compile_sky_light();
	void compile_texture_maps();
};

}

#endif // _PPM_PTRFREESCENE_H_
