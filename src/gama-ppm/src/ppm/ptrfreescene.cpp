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

	// clear vectors
	mesh_ids.resize(0);
	vertexes.resize(0);
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
		const lux_ext_mesh_list_t meshs = original_scene->objects;
		compile_mesh_first_triangle_offset(meshs);
		translate_geometry_qbvh(meshs);
	} else {
		translate_geometry();
//		throw string("Unsupported accelerator type ").append(config.accel_name);
	}

}

void PtrFreeScene :: compile_materials() {
	// reset all materials to false
	compiled_materials.resize(ppm::MAT_MAX);
	for(uint i = 0; i < compiled_materials.size(); ++i)
		compiled_materials[i] = false;

	const uint materials_count = original_scene->materials.size();
	materials.resize(materials_count);

	for(uint i = 0; i < materials_count; ++i) {
		const luxrays::Material* orig_m = original_scene->materials[i];
		ppm::Material* m = &materials[i];

		switch(orig_m->GetType()) {
		case luxrays::MATTE: {
			compiled_materials[ppm::MAT_MATTE] = true;
			const luxrays::MatteMaterial* mm = static_cast<const luxrays::MatteMaterial*>(orig_m);

			m->diffuse  = mm->IsDiffuse();
			m->specular = mm->IsSpecular();
			m->type = ppm::MAT_MATTE;
			m->param.matte.kd.r = mm->GetKd().r;
			m->param.matte.kd.g = mm->GetKd().g;
			m->param.matte.kd.b = mm->GetKd().b;
			break;
		}
		case luxrays::AREALIGHT: {
			compiled_materials[ppm::MAT_AREALIGHT] = true;
			const luxrays::AreaLightMaterial* alm = static_cast<const luxrays::AreaLightMaterial*>(orig_m);

			m->diffuse  = alm->IsDiffuse();
			m->specular = alm->IsSpecular();
			m->type = ppm::MAT_AREALIGHT;
			m->param.area_light.gain.r = alm->GetGain().r;
			m->param.area_light.gain.g = alm->GetGain().g;
			m->param.area_light.gain.b = alm->GetGain().b;
			break;
		}
		case luxrays::MIRROR: {
			compiled_materials[ppm::MAT_MIRROR] = true;
			const luxrays::MirrorMaterial* mm = static_cast<const luxrays::MirrorMaterial*>(orig_m);

			m->type = ppm::MAT_MIRROR;
			m->param.mirror.kr.r = mm->GetKr().r;
			m->param.mirror.kr.g = mm->GetKr().g;
			m->param.mirror.kr.b = mm->GetKr().b;
			m->param.mirror.specular_bounce = mm->HasSpecularBounceEnabled();
			break;
		}
		case luxrays::GLASS: {
			compiled_materials[ppm::MAT_GLASS] = true;
			const luxrays::GlassMaterial* gm = static_cast<const luxrays::GlassMaterial*>(orig_m);

			m->diffuse  = gm->IsDiffuse();
			m->specular = gm->IsSpecular();
			m->type = ppm::MAT_GLASS;
			m->param.glass.refl.r   = gm->GetKrefl().r;
			m->param.glass.refl.g   = gm->GetKrefl().g;
			m->param.glass.refl.b   = gm->GetKrefl().b;
			m->param.glass.refrct.r = gm->GetKrefrct().r;
			m->param.glass.refrct.g = gm->GetKrefrct().g;
			m->param.glass.refrct.b = gm->GetKrefrct().b;
			m->param.glass.outside_ior = gm->GetOutsideIOR();
			m->param.glass.R0 = gm->GetIOR();
			m->param.glass.reflection_specular_bounce   = gm->HasReflSpecularBounceEnabled();
			m->param.glass.transmission_specular_bounce = gm->HasRefrctSpecularBounceEnabled();
			break;
		}
		case luxrays::MATTEMIRROR: {
			compiled_materials[ppm::MAT_MATTEMIRROR] = true;
			const luxrays::MatteMirrorMaterial *mmm = static_cast<const luxrays::MatteMirrorMaterial*>(orig_m);

			m->diffuse  = mmm->IsDiffuse();
			m->specular = mmm->IsSpecular();
			m->type = ppm::MAT_MATTEMIRROR;
			m->param.matte_mirror.matte.kd.r  = mmm->GetMatte().GetKd().r;
			m->param.matte_mirror.matte.kd.g  = mmm->GetMatte().GetKd().g;
			m->param.matte_mirror.matte.kd.b  = mmm->GetMatte().GetKd().b;
			m->param.matte_mirror.mirror.kr.r = mmm->GetMirror().GetKr().r;
			m->param.matte_mirror.mirror.kr.g = mmm->GetMirror().GetKr().g;
			m->param.matte_mirror.mirror.kr.b = mmm->GetMirror().GetKr().b;
			m->param.matte_mirror.mirror.specular_bounce = mmm->GetMirror().HasSpecularBounceEnabled();
			m->param.matte_mirror.matte_filter = mmm->GetMatteFilter();
			m->param.matte_mirror.tot_filter = mmm->GetTotFilter();
			m->param.matte_mirror.matte_pdf = mmm->GetMattePdf();
			m->param.matte_mirror.mirror_pdf = mmm->GetMirrorPdf();
			break;
		}
		case luxrays::METAL: {
			compiled_materials[ppm::MAT_METAL] = true;
			const luxrays::MetalMaterial* mm = static_cast<const luxrays::MetalMaterial*>(orig_m);

			m->diffuse  = mm->IsDiffuse();
			m->specular = mm->IsSpecular();
			m->type = ppm::MAT_METAL;
			m->param.metal.kr.r = mm->GetKr().r;
			m->param.metal.kr.g = mm->GetKr().g;
			m->param.metal.kr.b = mm->GetKr().b;
			m->param.metal.exp = mm->GetExp();
			m->param.metal.specular_bounce = mm->HasSpecularBounceEnabled();
			break;
		}
		case luxrays::MATTEMETAL: {
			compiled_materials[ppm::MAT_MATTEMETAL] = true;
			const luxrays::MatteMetalMaterial* mmm = static_cast<const luxrays::MatteMetalMaterial*>(orig_m);

			m->diffuse  = mmm->IsDiffuse();
			m->specular = mmm->IsSpecular();
			m->type = ppm::MAT_METAL;
			m->param.matte_metal.matte.kd.r  = mmm->GetMatte().GetKd().r;
			m->param.matte_metal.matte.kd.g  = mmm->GetMatte().GetKd().g;
			m->param.matte_metal.matte.kd.b  = mmm->GetMatte().GetKd().b;
			m->param.matte_metal.metal.kr.r = mmm->GetMetal().GetKr().r;
			m->param.matte_metal.metal.kr.g = mmm->GetMetal().GetKr().g;
			m->param.matte_metal.metal.kr.b = mmm->GetMetal().GetKr().b;
			m->param.matte_metal.metal.exp = mmm->GetMetal().GetExp();
			m->param.matte_metal.metal.specular_bounce = mmm->GetMetal().HasSpecularBounceEnabled();
			m->param.matte_metal.matte_filter = mmm->GetMatteFilter();
			m->param.matte_metal.tot_filter = mmm->GetTotFilter();
			m->param.matte_metal.matte_pdf = mmm->GetMattePdf();
			m->param.matte_metal.metal_pdf = mmm->GetMetalPdf();
			break;
		}
		case luxrays::ALLOY: {
			compiled_materials[ppm::MAT_ALLOY] = true;
			const luxrays::AlloyMaterial* am = static_cast<const luxrays::AlloyMaterial*>(orig_m);

			m->diffuse  = am->IsDiffuse();
			m->specular = am->IsSpecular();
			m->type = ppm::MAT_ALLOY;
			m->param.alloy.refl.r  = am->GetKrefl().r;
			m->param.alloy.refl.g  = am->GetKrefl().g;
			m->param.alloy.refl.b  = am->GetKrefl().b;
			m->param.alloy.refl.r  = am->GetKd().r;
			m->param.alloy.refl.g  = am->GetKd().g;
			m->param.alloy.refl.b  = am->GetKd().b;
			m->param.alloy.exp = am->GetExp();
			m->param.alloy.R0 = am->GetR0();
			m->param.alloy.specular_bounce = am->HasSpecularBounceEnabled();
			break;
		}
		case luxrays::ARCHGLASS: {
			compiled_materials[ppm::MAT_ARCHGLASS] = true;
			const luxrays::ArchGlassMaterial* agm = static_cast<const luxrays::ArchGlassMaterial*>(orig_m);

			m->diffuse  = agm->IsDiffuse();
			m->specular = agm->IsSpecular();
			m->type = ppm::MAT_ARCHGLASS;
			m->param.arch_glass.refl.r   = agm->GetKrefl().r;
			m->param.arch_glass.refl.g   = agm->GetKrefl().g;
			m->param.arch_glass.refl.b   = agm->GetKrefl().b;
			m->param.arch_glass.refrct.r = agm->GetKrefrct().r;
			m->param.arch_glass.refrct.g = agm->GetKrefrct().g;
			m->param.arch_glass.refrct.b = agm->GetKrefrct().b;
			m->param.arch_glass.trans_filter = agm->GetTransFilter();
			m->param.arch_glass.tot_filter = agm->GetTotFilter();
			m->param.arch_glass.refl_pdf = agm->GetReflPdf();
			m->param.arch_glass.trans_pdf = agm->GetTransPdf();
			break;
		}
		default: /* MATTE */ {
			compiled_materials[ppm::MAT_MATTE] = true;
			m->type = ppm::MAT_MATTE;
			m->param.matte.kd.r = 0.75f;
			m->param.matte.kd.g = 0.75f;
			m->param.matte.kd.b = 0.75f;
			break;
		}
		}
	}

	// translate mesh material indexes
	const uint mesh_count = original_scene->objectMaterials.size();
	mesh_mats.resize(mesh_count);
	for(uint i = 0; i < mesh_count; ++i) {
		const luxrays::Material* m = original_scene->objectMaterials[i];

		// look for the index
		uint index = 0;
		for(uint j = 0; j < materials_count; ++j) {
			if (m == original_scene->materials[j]) {
				index = j;
				break;
			}
		}
		mesh_mats[i] = index;
	}

}

void PtrFreeScene :: compile_area_lights() {
	uint area_light_count;
	for(uint i = 0; i < original_scene->lights.size(); ++i) {
		if (original_scene->lights[i]->IsAreaLight())
			++area_light_count;
	}

	area_lights.resize(area_light_count);
	uint index = 0;
	if (area_light_count) {
		for(uint i = 0; i < original_scene->lights.size(); ++i) {
			if (original_scene->lights[i]->IsAreaLight()) {
				const luxrays::TriangleLight* tl = static_cast<const luxrays::TriangleLight*>(original_scene->lights[i]);
				const luxrays::ExtMesh* mesh = static_cast<const luxrays::ExtMesh*>(original_scene->objects[tl->GetMeshIndex()]);
				const luxrays::Triangle* tri = static_cast<const luxrays::Triangle*>(&mesh->GetTriangles()[tl->GetTriIndex()]);

				ppm::TriangleLight* cpl = &area_lights[index];
				cpl->v0 = ppm::Point(mesh->GetVertex(tri->v[0]));
				cpl->v1 = ppm::Point(mesh->GetVertex(tri->v[1]));
				cpl->v2 = ppm::Point(mesh->GetVertex(tri->v[2]));
				cpl->mesh_index = tl->GetMeshIndex();
				cpl->tri_index = tl->GetTriIndex();
				cpl->normal = mesh->GetNormal(tri->v[0]);
				cpl->area = tl->GetArea();

				const luxrays::AreaLightMaterial* alm = static_cast<const luxrays::AreaLightMaterial*>(tl->GetMaterial());
				cpl->gain = ppm::Spectrum(alm->GetGain());
				++index;
			}
		}
	}
}

void PtrFreeScene :: compile_infinite_light() {
	infinite_light_sp = smartPtr<ppm::InfiniteLight>(sizeof(ppm::InfiniteLight));

	const luxrays::InfiniteLight* il = NULL;
	if (original_scene->infiniteLight && ((original_scene->infiniteLight->GetType() == luxrays::TYPE_IL_BF)
			|| (original_scene->infiniteLight->GetType() == luxrays::TYPE_IL_PORTAL)
			|| (original_scene->infiniteLight->GetType() == luxrays::TYPE_IL_IS))) {
		il = original_scene->infiniteLight;
	} else {
		for(uint i = 0; i < original_scene->lights.size(); ++i) {
			const luxrays::LightSource* l = original_scene->lights[i];
			if ((l->GetType() == luxrays::TYPE_IL_BF)
					|| (l->GetType() == luxrays::TYPE_IL_PORTAL)
					|| (l->GetType() == luxrays::TYPE_IL_IS)) {
				il = static_cast<const luxrays::InfiniteLight*>(l);
				break;
			}
		}
	}

	if (il) {
		infinite_light_sp->exists = true;
		infinite_light_sp->gain   = ppm::Spectrum(il->GetGain());
		infinite_light_sp->shiftU = il->GetShiftU();
		infinite_light_sp->shiftV = il->GetShiftV();

		const luxrays::TextureMap* tex_map = il->GetTexture()->GetTexMap();
		infinite_light_sp->width  = tex_map->GetWidth();
		infinite_light_sp->height = tex_map->GetHeight();
	} else {
		infinite_light_sp->exists = false;
	}
}

void PtrFreeScene :: compile_sun_light() {
	sun_light_sp = smartPtr<ppm::SunLight>(sizeof(ppm::SunLight));

	const luxrays::SunLight* sl = NULL;
	for(uint i = 0; i < original_scene->lights.size(); ++i) {
		luxrays::LightSource* l = original_scene->lights[i];
		if (l->GetType() == luxrays::TYPE_SUN) {
			sl = static_cast<luxrays::SunLight*>(l);
			break;
		}
	}

	if (sl) {
		sun_light_sp->exists = true;
		sun_light_sp->gain   = ppm::Spectrum(sl->GetGain());
		sun_light_sp->turbidity = sl->GetTubidity();
		sun_light_sp->rel_size = sl->GetRelSize();
		float tmp;
		sun_light_sp->x = ppm::Vector(sl->x);
		sun_light_sp->y = ppm::Vector(sl->y);
		sun_light_sp->cos_theta_max = sl->cosThetaMax;
		sun_light_sp->color = ppm::Spectrum(sl->suncolor);
	} else {
		sun_light_sp->exists = false;
	}
}

void PtrFreeScene :: compile_sky_light() {
	sky_light_sp = smartPtr<SkyLight>(sizeof(ppm::SkyLight));

	const luxrays::SkyLight* sl = NULL;
	if (original_scene->infiniteLight
			&& (original_scene->infiniteLight->GetType() == luxrays::TYPE_IL_SKY)) {
		sl = static_cast<const luxrays::SkyLight*>(original_scene->infiniteLight);
	} else {
		for(uint i = 0; i < original_scene->lights.size(); ++i) {
			const luxrays::LightSource* l = original_scene->lights[i];
			if (l->GetType() == luxrays::TYPE_IL_SKY) {
				sl = static_cast<const luxrays::SkyLight*>(l);
				break;
			}
		}
	}

	if (sl) {
		sky_light_sp->exists = true;
		sky_light_sp->gain = ppm::Spectrum(sl->GetGain());
		sky_light_sp->theta_s = sl->thetaS;
		sky_light_sp->phi_s = sl->phiS;
		sky_light_sp->zenith_Y = sl->zenith_Y;
		sky_light_sp->zenith_x = sl->zenith_x;
		sky_light_sp->zenith_y = sl->zenith_y;
		for(uint i = 0; i < 6; ++i) {
			sky_light_sp->perez_Y[i] = sl->perez_Y[i];
			sky_light_sp->perez_x[i] = sl->perez_x[i];
			sky_light_sp->perez_y[i] = sl->perez_y[i];
		}
	} else {
		sky_light_sp->exists = false;
	}
}

void PtrFreeScene :: compile_texture_maps() {
	tex_maps.resize(0);
	rgb_tex.resize(0);
	alpha_tex.resize(0);
	mesh_texs.resize(0);
	bump_map.resize(0);
	bump_map_scales.resize(0);
	normal_map.resize(0);

	// translate mesh texture maps
	std::vector<luxrays::TextureMap*> tms;
	original_scene->texMapCache->GetTexMaps(tms);
	// compute amount of RAM to allocate
	uint rgb_tex_size = 0;
	uint alpha_tex_size = 0;
	for(uint i = 0; i < tms.size(); ++i) {
		luxrays::TextureMap* tm = tms[i];
		const uint pixel_count = tm->GetWidth() * tm->GetHeight();
		rgb_tex_size += pixel_count;
		if (tm->HasAlpha())
			alpha_tex_size += pixel_count;
	}

	// allocate texture map
	if ((rgb_tex_size > 0) || (alpha_tex_size) > 0) {
		tex_maps.resize(tms.size());

		if (rgb_tex_size > 0) {
			uint rgb_offset = 0;
			rgb_tex.resize(rgb_tex_size);
			for(uint i = 0; i < tms.size(); ++i) {
				luxrays::TextureMap* tm = tms[i];
				const uint pixel_count = tm->GetWidth() * tm->GetHeight();
				// TODO memcpy safe?
				memcpy(&rgb_tex[rgb_offset], tm->GetPixels(), pixel_count * sizeof(ppm::Spectrum));
				tex_maps[i].rgb_offset = rgb_offset;
				rgb_offset += pixel_count;
			}
		}

		if (alpha_tex_size > 0) {
			uint alpha_offset = 0;
			alpha_tex.resize(alpha_tex_size);
			for(uint i = 0; i < tms.size(); ++i) {
				luxrays::TextureMap* tm = tms[i];
				const uint pixel_count = tm->GetWidth() * tm->GetHeight();

				if (tm->HasAlpha()) {
					memcpy(&alpha_tex[alpha_offset], tm->GetAlphas(), pixel_count * sizeof(float));
					tex_maps[i].alpha_offset = alpha_offset;
					alpha_offset += pixel_count;
				} else {
					tex_maps[i].alpha_offset = PPM_NONE;
				}
			}
		}

		// translate texture map description
		for(uint i = 0; i < tms.size(); ++i) {
			luxrays::TextureMap* tm = tms[i];
			tex_maps[i].width = tm->GetWidth();
			tex_maps[i].height = tm->GetHeight();
		}

		// translate mesh texture indexes
		const uint mesh_count = mesh_mats.size();
		mesh_texs.resize(mesh_count);
		for(uint i = 0; i < mesh_count; ++i) {
			luxrays::TexMapInstance* t = original_scene->objectTexMaps[i];

			if (t) { // look for the index
				uint index = 0;
				for(uint j = 0; j < tms.size(); ++j) {
					if (t->GetTexMap() == tms[j]) {
						index = j;
						break;
					}
				}
				mesh_texs[i] = index;
			} else {
				mesh_texs[i] = PPM_NONE;
			}
		}

		// translate mesh bump map indexes
		bool has_bump_mapping = false;
		bump_map.resize(mesh_count);
		for(uint i = 0; i < mesh_count; +i) {
			luxrays::BumpMapInstance* bm = original_scene->objectBumpMaps[i];

			if (bm) { // look for the index
				uint index = 0;
				for(uint j = 0; j < tms.size(); ++j) {
					if (bm->GetTexMap() == tms[j]) {
						index = j;
						break;
					}
				}
				bump_map[i] = index;
				has_bump_mapping = true;
			} else {
				bump_map[i] = PPM_NONE;
			}
		}

		if (has_bump_mapping) {
			bump_map_scales.resize(mesh_count);
			for(uint i = 0; i < mesh_count; ++i) {
				luxrays::BumpMapInstance* bm = original_scene->objectBumpMaps[i];

				if (bm)
					bump_map_scales[i] = bm->GetScale();
				else
					bump_map_scales[i] = 1.f;
			}
		}

		// translate mesh normal map indices
		bool has_normal_mapping = false;
		normal_map.resize(mesh_count);
		for(uint i = 0; i < mesh_count; ++i) {
			luxrays::NormalMapInstance* nm = original_scene->objectNormalMaps[i];

			if (nm) { // look for the index
				uint index = 0;
				for(uint j = 0; j < tms.size(); ++j) {
					if (nm->GetTexMap() == tms[j]) {
						index = j;
						break;
					}
				}
				normal_map[i] = index;
				has_normal_mapping = true;
			} else {
				normal_map[i] = PPM_NONE;
			}
		}
	}
}

/*
 * auxiliary compilation methods
 */
void PtrFreeScene :: compile_mesh_first_triangle_offset(const lux_ext_mesh_list_t& meshs) {
	mesh_first_triangle_offset.resize(meshs.size());
	for(uint i = 0, current = 0; i < meshs.size(); ++i) {
		const luxrays::ExtMesh* mesh = meshs[i];
		mesh_first_triangle_offset[i] = current;
		current += mesh->GetTotalTriangleCount();
	}
}

void PtrFreeScene :: translate_geometry_qbvh(const lux_ext_mesh_list_t& meshs) {
	lux_defined_meshs_t defined_meshs(PtrFreeScene::mesh_ptr_compare);

	Mesh new_mesh;
	Mesh current_mesh;

	for(lux_ext_mesh_list_t::const_iterator it = meshs.begin(); it != meshs.end(); ++it) {
		const luxrays::ExtMesh* mesh = *it;
		bool is_existing_instance;
		if (mesh->GetType() == luxrays::TYPE_EXT_TRIANGLE_INSTANCE) {
			const luxrays::ExtInstanceTriangleMesh* imesh = static_cast<const luxrays::ExtInstanceTriangleMesh*>(mesh);

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

			const luxrays::Transform trans = imesh->GetTransformation();
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
			vertexes.resize(mesh->GetTotalVertexCount());
			for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
				vertexes[j] = ppm::Point(mesh->GetVertex(j));

			// translate mesh indices
			const luxrays::Triangle *mtris = mesh->GetTriangles();
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
		const luxrays::ExtMesh* mesh = original_scene->objects[i];

		mesh_offsets[i] = v_index;
		for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j) {
			normals[index]  = ppm::Normal(mesh->GetNormal(j));
			colors[index]   = ppm::Spectrum(mesh->GetColor(j));
			uvs[index]      = (mesh->HasUVs()) ? ppm::UV(mesh->GetUV(j)) : ppm::UV((0.f, 0.f));
			vertexes[index] = ppm::Point(mesh->GetVertex(j));
			index++;
		}
		v_index += mesh->GetTotalVertexCount();
	}

	// translate mesh triangles
	triangles.resize(n_triangles);
	index = 0;
	for(uint i = 0; i < original_scene->objects.size(); ++i) {
		const luxrays::ExtMesh* mesh   = original_scene->objects[i];
		const luxrays::Triangle *mtris = mesh->GetTriangles();
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

ostream& operator<< (ostream& os, PtrFreeScene& scene) {
	os << "Vertexes:\n\t";
	for(uint i(0); i < scene.vertexes.size(); ++i)
		os << ' ' << scene.vertexes[i];

	os <<  "\n\nNormals:\n\t";
	for(uint i(0); i < scene.normals.size(); ++i)
		os << ' ' << scene.normals[i];

	os <<  "\n\nColors:\n\t";
	for(uint i(0); i < scene.colors.size(); ++i)
		os << ' ' << scene.colors[i];

	os << "\n\nUVs:\n\t";
	for(uint i(0); i < scene.uvs.size(); ++i)
		os << ' ' << scene.uvs[i];

	os << "\n\nTriangles:\n\t";
	for(uint i(0); i < scene.triangles.size(); ++i)
		os << ' ' << scene.triangles[i];

	os << "\n\nMeshDescs:\n\t";
	for(uint i(0); i < scene.mesh_descs.size(); ++i)
		os << ' ' << scene.mesh_descs[i];

	os << "\n\nMeshIDs:\n\t";
	for(uint i(0); i < scene.mesh_ids.size(); ++i)
		os << ' ' << scene.mesh_ids[i];

	os << "\n\nMeshFirstTriangleOffset:\n\t";
	for(uint i(0); i < scene.mesh_first_triangle_offset.size(); ++i)
		os << ' ' << scene.mesh_first_triangle_offset[i];

	os << "\n\nBSphere:\n\t" << scene.bsphere_sp[0];
	os << "\n\nCamera:\n\t" << scene.camera_sp[0];

	os << "\n\nCompiledMaterials:\n\t";
	for(uint i(0); i < scene.compiled_materials.size(); ++i)
		os << ' ' << scene.compiled_materials[i];

	os << "\n\nMaterials:\n\t";
	for(uint i(0); i < scene.materials.size(); ++i)
		os << ' ' << scene.materials[i];

	os << "\n\nMeshMaterials:\n\t";
	for(uint i(0); i < scene.mesh_mats.size(); ++i)
		os << ' ' << scene.mesh_mats[i];

	os << "\n\nAreaLights:\n\t";
	for(uint i(0); i < scene.area_lights.size(); ++i)
		os << ' ' << scene.area_lights[i];

	os << "\n\nInfiniteLight:\n\t" << scene.infinite_light_sp[0];
	os << "\n\nSunLight:\n\t" << scene.sun_light_sp[0];
	os << "\n\nSkyLight:\n\t" << scene.sky_light_sp[0];

	os << "\n\nTexMaps:\n\t";
	for(uint i(0); i < scene.tex_maps.size(); ++i)
		os << ' ' << scene.tex_maps[i];

	os << "\n\nRGBTex:\n\t";
	for(uint i(0); i < scene.rgb_tex.size(); ++i)
		os << ' ' << scene.rgb_tex[i];

	os << "\n\nAlphaTex:\n\t";
	for(uint i(0); i < scene.alpha_tex.size(); ++i)
		os << ' ' << scene.alpha_tex[i];

	os << "\n\nMeshTexs:\n\t";
	for(uint i(0); i < scene.mesh_texs.size(); ++i)
		os << ' ' << scene.mesh_texs[i];

	os << "\n\nBumpMap:\n\t";
	for(uint i(0); i < scene.bump_map.size(); ++i)
		os << ' ' << scene.bump_map[i];

	os << "\n\nBumpMapScales:\n\t";
	for(uint i(0); i < scene.bump_map_scales.size(); ++i)
		os << ' ' << scene.bump_map_scales[i];

	os << "\n\nNormalMap:\n\t";
	for(uint i(0); i < scene.normal_map.size(); ++i)
		os << ' ' << scene.normal_map[i];

	return os;
}

}
