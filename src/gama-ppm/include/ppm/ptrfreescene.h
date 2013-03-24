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
#include "luxrays/utils/sdl/scene.h"
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

	typedef std::vector<luxrays::ExtMesh*> lux_ext_mesh_list_t;
	typedef bool(*lux_mesh_comparator_t)(luxrays::Mesh*, luxrays::Mesh*);
	typedef std::map<luxrays::ExtMesh*, uint, lux_mesh_comparator_t> lux_defined_meshs_t;

private:
	const Config& config;           // reference to global configs
	luxrays::Scene* original_scene; // original scene structure
	luxrays::DataSet* data_set;     // original data_set structure

	gama::vector<Point>    vertexes;
	gama::vector<Normal>   normals;
	gama::vector<Spectrum> colors;
	gama::vector<UV>       uvs;
	gama::vector<Triangle> triangles;
	gama::vector<Mesh>     mesh_descs;

	gama::vector<uint> mesh_ids;
	//const uint* mesh_ids;

	gama::vector<uint> mesh_first_triangle_offset;
	smartPtr<BSphere> bsphere_sp; // bounding sphere of the scene
	smartPtr<Camera> camera_sp;   // compiled camera

	// materials
	gama::vector<bool>     compiled_materials;
	gama::vector<Material> materials;
	gama::vector<uint>     mesh_mats;

	// lights
	gama::vector<TriangleLight> area_lights;
	smartPtr<InfiniteLight>     infinite_light_sp;
	smartPtr<SunLight>          sun_light_sp;
	smartPtr<SkyLight>          sky_light_sp;

	// textures
	gama::vector<TexMap> tex_maps;
	gama::vector<Spectrum> rgb_tex;
	gama::vector<float> alpha_tex;
	gama::vector<uint> mesh_texs;

	// bump maps
	gama::vector<uint> bump_map;
	gama::vector<float> bump_map_scales;

	// normal maps
	gama::vector<uint> normal_map;


	void compile_camera();
	void compile_geometry();
	void compile_materials();
	void compile_area_lights();
	void compile_infinite_light();
	void compile_sun_light();
	void compile_sky_light();
	void compile_texture_maps();

	// auxiliary compilation methods
	void compile_mesh_first_triangle_offset(const lux_ext_mesh_list_t& meshs);
	void translate_geometry();
	void translate_geometry_qbvh(const lux_ext_mesh_list_t& meshs);

	friend ostream& operator<< (ostream& os, PtrFreeScene& scene);
public:
//	static lux_mesh_comparator_t mesh_ptr_compare;
	static bool mesh_ptr_compare(luxrays::Mesh* m0, luxrays::Mesh* m1);

};

}

#endif // _PPM_PTRFREESCENE_H_
