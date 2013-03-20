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

enum {
	MAT_MATTE,
	MAT_AREALIGHT,
	MAT_MIRROR,
	MAT_GLASS,
	MAT_MATTEMIRROR,
	MAT_METAL,
	MAT_MATTEMETAL,
	MAT_ALLOY,
	MAT_ARCHGLASS,

	MAT_MAX
} CompiledMaterials_e;


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

	gama::vector<Point>    vertices;
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
	smartPtr<InfiniteLight>     infinite_light;
	smartPtr<SunLight>          sun_light;
	smartPtr<SkyLight>          sky_light;

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
	void compile_mesh_first_triangle_offset(lux_ext_mesh_list_t& meshs);
	void translate_geometry();
	void translate_geometry_qbvh(lux_ext_mesh_list_t& meshs);
public:
//	static lux_mesh_comparator_t mesh_ptr_compare;
	static bool mesh_ptr_compare(luxrays::Mesh* m0, luxrays::Mesh* m1);

};

}

#endif // _PPM_PTRFREESCENE_H_
