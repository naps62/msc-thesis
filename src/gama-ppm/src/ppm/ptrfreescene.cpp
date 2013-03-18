/*
 * ptrfreescene.cpp
 *
 *  Created on: Mar 11, 2013
 *      Author: Miguel Palhas
 */

#include "ppm/ptrfreescene.h"

#include "luxrays/core/accelerator.h"

namespace ppm {

//
// constructors
//

// constructor receiving a config struct
//*************************
PtrFreeScene :: PtrFreeScene(const Config& _config)
: config(_config) {
//************************
	// load input scene in luxrays format
	// TODO what is this -1? Is it the accelerator structure?
	original_scene = new luxrays::Scene(config.scene_file, config.width, config.height, config.accel_type);
	data_set = original_scene->UpdateDataSet();

	// recompile the entire scene
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
	camera_sp(sizeof(ppm::Camera));
	luxrays::PerspectiveCamera original = *(original_scene->camera);
	camera_sp->compile(original);
}

void PtrFreeScene :: compile_geometry() {
//	const uint n_vertices  = data_set->GetTotalVertexCount();
//	const uint n_triangles = data_set->GetTotalTriangleCount();

	// clear vectors
	mesh_ids.resize(0);
	vertices.resize(0);
	normals.resize(0);
	colors.resize(0);
	uvs.resize(0);
	triangles.resize(0);
	mesh_descs.resize(0);
	mesh_first_triangle_offset.resize(0);
	bsphere_sp(sizeof(ppm::BSphere));

	// copy mesh_id_table
	// TODO check if this is valid
	uint* original_mesh_ids = data_set->GetMeshIDTable();
	mesh_ids.resize(data_set->GetTotalTriangleCount());
	for(uint i = 0; i < data_set->GetTotalTriangleCount(); ++i)
		mesh_ids[i] = original_mesh_ids[i]; // TODO probably change this to a memcpy

	// get scene bsphere
	bsphere_sp[0] = data_set->GetPPMBSphere();

	// check used accelerator type
	if (config.accel_type == ppm::ACCEL_QBVH) {
		lux_ext_mesh_list_t meshs = original_scene->objects;
		compile_mesh_first_triangle_offset(meshs);
		translate_geometry_qbvh(meshs);
	} else {
		translate_geometry();
		throw string("Unsupported accelerator type ").append(config.accel_name);
	}

}

void PtrFreeScene :: compile_materials() {
	// TODO
}

void PtrFreeScene :: compile_area_lights() {
	// TODO
}

void PtrFreeScene :: compile_infinite_light() {
	// TODO
}

void PtrFreeScene :: compile_sun_light() {
	// TODO
}

void PtrFreeScene :: compile_sky_light() {
	// TODO
}

void PtrFreeScene :: compile_texture_maps() {
	// TODO
}

/*
 * auxiliary compilation methods
 */
void PtrFreeScene :: compile_mesh_first_triangle_offset(lux_ext_mesh_list_t& meshs) {
	mesh_first_triangle_offset.resize(meshs.size());
	for(uint i = 0, current = 0; i < meshs.size(); ++i) {
		luxrays::ExtMesh* mesh = meshs[i];
		mesh_first_triangle_offset[i] = current;
		current += mesh->GetTotalTriangleCount();
	}
}

void PtrFreeScene :: translate_geometry_qbvh(lux_ext_mesh_list_t& meshs) {
	lux_defined_meshs_t defined_meshs(PtrFreeScene::mesh_ptr_compare);

	Mesh new_mesh;
	Mesh current_mesh;

	for(lux_ext_mesh_list_t::iterator it = meshs.begin(); it != meshs.end(); ++it) {
		luxrays::ExtMesh* mesh = *it;
		bool is_existing_instance;
		if (mesh->GetType() == luxrays::TYPE_EXT_TRIANGLE_INSTANCE) {
			luxrays::ExtInstanceTriangleMesh* imesh = static_cast<luxrays::ExtInstanceTriangleMesh*>(mesh);

			// check if is one of the already done meshes
			lux_defined_meshs_t::iterator it = defined_meshs.find(imesh->GetExtTriangleMesh());
			if (it == defined_meshs.end()) {
				// it is a new one
				current_mesh = new_mesh;

				new_mesh.verts_offset += imesh->GetTotalVertexCount();
				new_mesh.tris_offset  += imesh->GetTotalTriangleCount();
				is_existing_instance = false;
				const uint index = mesh_descs.size();
				defined_meshs[imesh->GetExtTriangleMesh()] = index;
			} else {
				// it is not a new one
				current_mesh = mesh_descs[it->second];
				is_existing_instance = true;
			}

			luxrays::Transform trans = imesh->GetTransformation();
			current_mesh.trans.set(imesh->GetTransformation().GetMatrix().m);
			current_mesh.inv_trans.set(imesh->GetInvTransformation().GetMatrix().m);

			mesh = imesh->GetExtTriangleMesh();
		} else {
			// not a luxrays::TYPE_EXT_TRIANGLE_INSTANCE
			current_mesh = new_mesh;
			new_mesh.verts_offset += mesh->GetTotalVertexCount();
			new_mesh.tris_offset  += mesh->GetTotalTriangleCount();

			if (mesh->HasColors()) {
				new_mesh.colors_offset += mesh->GetTotalVertexCount();
				current_mesh.has_colors = true;
			} else {
				current_mesh.has_colors = false;
			}
			is_existing_instance = false;
		}

		mesh_descs.push_back(current_mesh);

		if (!is_existing_instance) {

			assert(mesh->GetType() == luxrays::TYPE_EXT_TRIANGLE);

			// translate mesh normals and colors

			normals.resize(mesh->GetTotalVertexCount());
			for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
				normals[j] = ppm::Normal(mesh->GetNormal(j));

			if (mesh->HasColors()) {
				colors.resize(mesh->GetTotalVertexCount());
				for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
					colors[j] = ppm::Spectrum(mesh->GetColor(j));
			}

			// translate vertex uvs

			if (original_scene->texMapCache->GetSize()) {
				// TODO: should check if the only texture map is used for infintelight
				uvs.resize(mesh->GetTotalVertexCount());
				if (mesh->HasUVs())
					for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
						uvs[j] = ppm::UV(0.f, 0.f);
				else
					for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
						uvs[j] = ppm::UV(mesh->GetUV(j));
			}

			// translate mesh vertices
			vertices.resize(mesh->GetTotalVertexCount());
			for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
				vertices[j] = ppm::Point(mesh->GetVertex(j));

			// translate mesh indices
			luxrays::Triangle *mtris = mesh->GetTriangles();
			triangles.resize(mesh->GetTotalTriangleCount());
			for(uint j = 0; j < mesh->GetTotalTriangleCount(); ++j)
				triangles[j] = ppm::Triangle(mtris[j]);
		}
	}
}

void PtrFreeScene :: translate_geometry() {
	mesh_first_triangle_offset.resize(0);
	const uint n_vertices  = data_set->GetTotalVertexCount();
	const uint n_triangles = data_set->GetTotalTriangleCount();

	// translate mesh normals, colors and uvs
	normals.resize(n_vertices);
	colors.resize(n_vertices);
	uvs.resize(n_vertices);
	uint index = 0;

	// aux data to later translate triangles
	uint *mesh_offsets = new uint[original_scene->objects.size()];
	uint v_index = 0;

	for (uint i = 0; i < original_scene->objects.size(); ++i) {
		luxrays::ExtMesh* mesh = original_scene->objects[i];

		mesh_offsets[i] = v_index;
		for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j) {
			normals[index]  = ppm::Normal(mesh->GetNormal(j));
			colors[index]   = ppm::Spectrum(mesh->GetColor(j));
			uvs[index]      = (mesh->HasUVs()) ? ppm::UV(mesh->GetUV(j)) : ppm::UV((0.f, 0.f));
			vertices[index] = ppm::Point(mesh->GetVertex(j));
			index++;
		}
		v_index += mesh->GetTotalVertexCount();
	}

	// translate mesh triangles
	triangles.resize(n_triangles);
	index = 0;
	for(uint i = 0; i < original_scene->objects.size(); ++i) {
		luxrays::ExtMesh* mesh   = original_scene->objects[i];
		luxrays::Triangle *mtris = mesh->GetTriangles();
		const uint moffset = mesh_offsets[i];
		for (uint j = 0; j < mesh->GetTotalTriangleCount(); ++j) {
			triangles[index++] = ppm::Triangle(
					mtris[j].v[0] + moffset,
					mtris[j].v[1] + moffset,
					mtris[j].v[2] + moffset);
		}
	}

	delete[] mesh_offsets;
}

bool PtrFreeScene :: mesh_ptr_compare(luxrays::Mesh* m0, luxrays::Mesh* m1) {
	return m0 < m1;
}

}
