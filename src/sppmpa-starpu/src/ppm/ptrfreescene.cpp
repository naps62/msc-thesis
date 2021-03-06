#include "ppm/ptrfreescene.h"

#include "luxrays/core/accelerator.h"
#include "ppm/math.h"

using namespace std;

namespace ppm {

//
// constructors
//

PtrFreeScene :: PtrFreeScene() {
}

// constructor receiving a config struct
//*************************
PtrFreeScene :: PtrFreeScene(const Config& config)
: accel_type(config.accel_type) {
//************************
  // load input scene in luxrays format
  // TODO what is this -1? Is it the accelerator structure?
  //data_set = original_scene->UpdateDataSet();
  original_scene = new luxrays::Scene(config.scene_file, config.width, config.height, config.accel_type);
  data_set = original_scene->UpdateDataSet();

  vertexes = NULL;
  normals = NULL;
  colors = NULL;
  uvs = NULL;
  triangles = NULL;
  mesh_descs = NULL;
  mesh_ids = NULL;
  mesh_first_triangle_offset = NULL;
  compiled_materials = NULL;
  materials = NULL;
  mesh_materials = NULL;
  area_lights = NULL;
  tex_maps = NULL;
  rgb_tex = NULL;
  alpha_tex = NULL;
  mesh_texs = NULL;
  bump_map = NULL;
  bump_map_scales = NULL;
  normal_map = NULL;

  // recompile the entire scene
  ActionList actions;
  actions.add_all();
  recompile(actions);

  n_nodes = data_set->GetAccelerator()->GetNodesCount();
  n_prims = data_set->GetAccelerator()->GetPrimsCount();
  nodes = NULL;
  prims = NULL;
}

PtrFreeScene :: ~PtrFreeScene() {
  delete original_scene;
  delete data_set;
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

bool PtrFreeScene :: intersect(Ray& ray, RayHit& hit) const {
  return data_set->Intersect(&ray, &hit);
}

/*
 * private methods
 */

void PtrFreeScene :: compile_camera() {
  luxrays::PerspectiveCamera original = *(original_scene->camera);
  camera.compile(original);
}

void PtrFreeScene :: compile_geometry() {

  // clear vectors
  delete_array(mesh_ids);
  delete_array(vertexes);
  delete_array(normals);
  delete_array(colors);
  delete_array(uvs);
  delete_array(triangles);
  delete_array(mesh_descs);
  delete_array(mesh_first_triangle_offset);

  this->mesh_count = original_scene->objects.size();//data_set->meshes.size();

  // copy mesh_id_table
  // TODO check if this is valid
  //mesh_ids.resize(data_set->meshes.size());
  this->mesh_ids = new uint[data_set->totalTriangleCount];
  //this->mesh_ids = data_set->GetMeshIDTable();
  for(unsigned i = 0; i < data_set->totalTriangleCount; ++i)
    this->mesh_ids[i] = data_set->GetMeshIDTable()[i]; // TODO probably change this to a memcpy

  // get scene bsphere
  this->bsphere = data_set->GetPPMBSphere();

  // check used accelerator type
  if (accel_type == ppm::ACCEL_QBVH) {
    const lux_ext_mesh_list_t meshs = original_scene->objects;
    compile_mesh_first_triangle_offset(meshs);
    translate_geometry_qbvh(meshs);
  } else {
    cout << "here" << endl;
    translate_geometry();
//    throw string("Unsupported accelerator type ").append(config.accel_name);
  }

}

void PtrFreeScene :: compile_materials() {
  // reset all materials to false
  //compiled_materials.resize(ppm::MAT_NULL);
  this->compiled_materials_count = ppm::MAT_MAX;
  reset_array(this->compiled_materials, this->compiled_materials_count);
  for(uint i = 0; i < ppm::MAT_MAX; ++i)
    this->compiled_materials[i] = false;

  this->materials_count = original_scene->materials.size();
  //materials.resize(materials_count);
  reset_array(this->materials, this->materials_count);
  memset(this->materials, 0, sizeof(ppm::Material)*this->materials_count);

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
      m->param.glass.ior = gm->GetIOR();
      m->param.glass.R0 = gm->GetR0();
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
      m->type = ppm::MAT_MATTEMETAL;
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
      m->param.alloy.diff.r  = am->GetKd().r;
      m->param.alloy.diff.g  = am->GetKd().g;
      m->param.alloy.diff.b  = am->GetKd().b;
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
  this->mesh_materials_count = original_scene->objectMaterials.size();
  //mesh_mats.resize(mesh_coutlt);
  reset_array(this->mesh_materials, this->mesh_materials_count);
  for(uint i = 0; i < this->mesh_materials_count; ++i) {
    const luxrays::Material* m = original_scene->objectMaterials[i];

    // look for the index
    uint index = 0;
    for(uint j = 0; j < materials_count; ++j) {
      if (m == original_scene->materials[j]) {
        index = j;
        break;
      }
    }
    this->mesh_materials[i] = index;
  }

}

void PtrFreeScene :: compile_area_lights() {
  this->area_lights_count = 0;
  for(uint i = 0; i < original_scene->lights.size(); ++i) {
    if (original_scene->lights[i]->IsAreaLight())
      ++this->area_lights_count;
  }

  //area_lights.resize(area_lights_count);
  reset_array(this->area_lights, this->area_lights_count);
  uint index = 0;
  if (this->area_lights_count) {
    for(uint i = 0; i < original_scene->lights.size(); ++i) {
      if (original_scene->lights[i]->IsAreaLight()) {
        const luxrays::TriangleLight* tl = static_cast<const luxrays::TriangleLight*>(original_scene->lights[i]);
        const luxrays::ExtMesh* mesh = static_cast<const luxrays::ExtMesh*>(original_scene->objects[tl->GetMeshIndex()]);
        const luxrays::Triangle* tri = static_cast<const luxrays::Triangle*>(&mesh->GetTriangles()[tl->GetTriIndex()]);

        ppm::TriangleLight* cpl = &area_lights[index];
        cpl->v0 = Point(mesh->GetVertex(tri->v[0]));
        cpl->v1 = Point(mesh->GetVertex(tri->v[1]));
        cpl->v2 = Point(mesh->GetVertex(tri->v[2]));
        cpl->mesh_index = tl->GetMeshIndex();
        cpl->tri_index = tl->GetTriIndex();
        cpl->normal = mesh->GetNormal(tri->v[0]);
        cpl->area = tl->GetArea();

        const luxrays::AreaLightMaterial* alm = static_cast<const luxrays::AreaLightMaterial*>(tl->GetMaterial());
        cpl->gain = Spectrum(alm->GetGain());
        ++index;
      }
    }
  }
}

void PtrFreeScene :: compile_infinite_light() {

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
    infinite_light.exists = true;
    infinite_light.gain   = Spectrum(il->GetGain());
    infinite_light.shiftU = il->GetShiftU();
    infinite_light.shiftV = il->GetShiftV();

    const luxrays::TextureMap* tex_map = il->GetTexture()->GetTexMap();
    infinite_light.width  = tex_map->GetWidth();
    infinite_light.height = tex_map->GetHeight();

    infinite_light_map = new Spectrum[infinite_light.width * infinite_light.height];
    for(unsigned i = 0; i < infinite_light.width * infinite_light.height; ++i) {
      infinite_light_map[i] = tex_map->GetPixels()[i];
    }
  } else {
    infinite_light.exists = false;
  }
}

void PtrFreeScene :: compile_sun_light() {

  const luxrays::SunLight* sl = NULL;
  for(uint i = 0; i < original_scene->lights.size(); ++i) {
    luxrays::LightSource* l = original_scene->lights[i];
    if (l->GetType() == luxrays::TYPE_SUN) {
      sl = static_cast<luxrays::SunLight*>(l);
      break;
    }
  }

  if (sl) {
    sun_light.exists = true;
    sun_light.gain   = Spectrum(sl->GetGain());
    sun_light.turbidity = sl->GetTubidity();
    sun_light.rel_size = sl->GetRelSize();
    sun_light.x = Vector(sl->x);
    sun_light.y = Vector(sl->y);
    sun_light.cos_theta_max = sl->cosThetaMax;
    sun_light.color = Spectrum(sl->suncolor);
  } else {
    sun_light.exists = false;
  }
}

void PtrFreeScene :: compile_sky_light() {

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
    sky_light.exists = true;
    sky_light.gain = Spectrum(sl->GetGain());
    sky_light.theta_s = sl->thetaS;
    sky_light.phi_s = sl->phiS;
    sky_light.zenith_Y = sl->zenith_Y;
    sky_light.zenith_x = sl->zenith_x;
    sky_light.zenith_y = sl->zenith_y;
    for(uint i = 0; i < 6; ++i) {
      sky_light.perez_Y[i] = sl->perez_Y[i];
      sky_light.perez_x[i] = sl->perez_x[i];
      sky_light.perez_y[i] = sl->perez_y[i];
    }
  } else {
    sky_light.exists = false;
  }
}

void PtrFreeScene :: compile_texture_maps() {
  delete_array(tex_maps);
  delete_array(rgb_tex);
  delete_array(alpha_tex);
  delete_array(mesh_texs);
  delete_array(bump_map);
  delete_array(bump_map_scales);
  delete_array(normal_map);
  this->tex_maps_count = 0;

  // translate mesh texture maps
  std::vector<luxrays::TextureMap*> tms;
  original_scene->texMapCache->GetTexMaps(tms);
  // compute amount of RAM to allocate
  //uint rgb_tex_size = 0;
  //uint alpha_tex_size = 0;
  this->rgb_tex_count = 0;
  this->alpha_tex_count = 0;
  for(uint i = 0; i < tms.size(); ++i) {
    luxrays::TextureMap* tm = tms[i];
    const uint pixel_count = tm->GetWidth() * tm->GetHeight();
    this->rgb_tex_count += pixel_count;
    if (tm->HasAlpha())
      this->alpha_tex_count += pixel_count;
  }

  // allocate texture map
  if ((this->rgb_tex_count > 0) || (this->alpha_tex_count) > 0) {
    this->tex_maps_count = tms.size();
    reset_array(this->tex_maps, this->tex_maps_count);
    //tex_maps.resize(tms.size());

    if (this->rgb_tex_count > 0) {
      uint rgb_offset = 0;
      //rgb_tex.resize(rgb_tex_size);
      reset_array(this->rgb_tex, this->rgb_tex_count);
      for(uint i = 0; i < tms.size(); ++i) {
        luxrays::TextureMap* tm = tms[i];
        const uint pixel_count = tm->GetWidth() * tm->GetHeight();
        // TODO memcpy safe?
        memcpy(&rgb_tex[rgb_offset], tm->GetPixels(), pixel_count * sizeof(Spectrum));
        this->tex_maps[i].rgb_offset = rgb_offset;
        rgb_offset += pixel_count;
      }
    }

    if (this->alpha_tex_count > 0) {
      uint alpha_offset = 0;
      //alpha_tex.resize(alpha_tex_size);
      reset_array(this->alpha_tex, this->alpha_tex_count);
      for(uint i = 0; i < tms.size(); ++i) {
        luxrays::TextureMap* tm = tms[i];
        const uint pixel_count = tm->GetWidth() * tm->GetHeight();

        if (tm->HasAlpha()) {
          memcpy(&alpha_tex[alpha_offset], tm->GetAlphas(), pixel_count * sizeof(float));
          this->tex_maps[i].alpha_offset = alpha_offset;
          alpha_offset += pixel_count;
        } else {
          this->tex_maps[i].alpha_offset = PPM_NONE;
        }
      }
    }

    // translate texture map description
    for(uint i = 0; i < tms.size(); ++i) {
      luxrays::TextureMap* tm = tms[i];
      this->tex_maps[i].width = tm->GetWidth();
      this->tex_maps[i].height = tm->GetHeight();
    }

    // translate mesh texture indexes
    //const uint mesh_count = mesh_mats.size();
    //mesh_texs.resize(mesh_count);
    reset_array(this->mesh_texs, this->mesh_materials_count);
    for(uint i = 0; i < this->mesh_materials_count; ++i) {
      luxrays::TexMapInstance* t = original_scene->objectTexMaps[i];

      if (t) { // look for the index
        uint index = 0;
        for(uint j = 0; j < tms.size(); ++j) {
          if (t->GetTexMap() == tms[j]) {
            index = j;
            break;
          }
        }
        this->mesh_texs[i] = index;
      } else {
        this->mesh_texs[i] = PPM_NONE;
      }
    }

    // translate mesh bump map indexes
    bool has_bump_mapping = false;
    //bump_map.resize(mesh_count);
    reset_array(this->bump_map, this->mesh_materials_count);
    for(uint i = 0; i < this->mesh_materials_count; ++i) {
      luxrays::BumpMapInstance* bm = original_scene->objectBumpMaps[i];

      if (bm) { // look for the index
        uint index = 0;
        for(uint j = 0; j < tms.size(); ++j) {
          if (bm->GetTexMap() == tms[j]) {
            index = j;
            break;
          }
        }
        this->bump_map[i] = index;
        has_bump_mapping = true;
      } else {
        this->bump_map[i] = PPM_NONE;
      }
    }

    if (has_bump_mapping) {
      //bump_map_scales.resize(mesh_count);
      reset_array(this->bump_map_scales, this->mesh_materials_count);
      for(uint i = 0; i < mesh_count; ++i) {
        luxrays::BumpMapInstance* bm = original_scene->objectBumpMaps[i];

        if (bm)
          this->bump_map_scales[i] = bm->GetScale();
        else
          this->bump_map_scales[i] = 1.f;
      }
    }

    // translate mesh normal map indices
    //unused? bool has_normal_mapping = false;
    //normal_map.resize(mesh_count);
    reset_array(this->normal_map, this->mesh_materials_count);
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
        this->normal_map[i] = index;
        //has_normal_mapping = true;
      } else {
        this->normal_map[i] = PPM_NONE;
      }
    }
  }
}

/*
 * auxiliary compilation methods
 */
void PtrFreeScene :: compile_mesh_first_triangle_offset(const lux_ext_mesh_list_t& meshs) {
  //mesh_first_triangle_offset.resize(meshs.size());
  reset_array(this->mesh_first_triangle_offset, this->mesh_count);
  for(uint i = 0, current = 0; i < original_scene->objects.size(); ++i) {
    const luxrays::ExtMesh* mesh = meshs[i];
    mesh_first_triangle_offset[i] = current;
    current += mesh->GetTotalTriangleCount();
  }
}

void PtrFreeScene :: translate_geometry_qbvh(const lux_ext_mesh_list_t& meshs) {
  lux_defined_meshs_t defined_meshs(PtrFreeScene::mesh_ptr_compare);

  Mesh new_mesh;
  Mesh current_mesh;

  std::vector<Point>    tmp_vertexes;
  std::vector<Normal>   tmp_normals;
  std::vector<Spectrum> tmp_colors;
  std::vector<UV>       tmp_uvs;
  std::vector<Triangle> tmp_triangles;
  std::vector<Mesh>     tmp_mesh_descs;

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
        const uint index = tmp_mesh_descs.size();
        defined_meshs[imesh->GetExtTriangleMesh()] = index;
      } else {
        // it is not a new one
        current_mesh = tmp_mesh_descs[it->second];
        is_existing_instance = true;
      }

      current_mesh.trans = imesh->GetTransformation().GetMatrix().m;
      current_mesh.inv_trans = imesh->GetInvTransformation().GetMatrix().m;

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

    tmp_mesh_descs.push_back(current_mesh);

    if (!is_existing_instance) {

      assert(mesh->GetType() == luxrays::TYPE_EXT_TRIANGLE);

      // translate mesh normals and colors

      uint offset = tmp_normals.size();
      tmp_normals.resize(offset + mesh->GetTotalVertexCount());
      for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
        tmp_normals[offset + j] = Normal(mesh->GetNormal(j));

      if (mesh->HasColors()) {
        offset = tmp_colors.size();
        tmp_colors.resize(offset + mesh->GetTotalVertexCount());
        for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
          tmp_colors[offset + j] = Spectrum(mesh->GetColor(j));
      }

      // translate vertex uvs

      if (original_scene->texMapCache->GetSize()) {
        // TODO: should check if the only texture map is used for infintelight
        offset = tmp_uvs.size();
        tmp_uvs.resize(offset + mesh->GetTotalVertexCount());
        if (mesh->HasUVs())
          for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
            tmp_uvs[offset + j] = UV(mesh->GetUV(j));
        else
          for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
            tmp_uvs[offset + j] = UV(0.f, 0.f);
      }

      // translate meshrverticener size, the content is expanded by inserting at the end as many elements as needed to reach a size of n. If val is specified, the new elements are initialized as copies of val, otherwise, they are value-initialized.s
      offset = tmp_vertexes.size();
      tmp_vertexes.resize(offset + mesh->GetTotalVertexCount());
      for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j)
        tmp_vertexes[offset + j] = Point(mesh->GetVertex(j));

      // translate mesh indices
      offset = tmp_triangles.size();
      const luxrays::Triangle *mtris = mesh->GetTriangles();
      tmp_triangles.resize(offset + mesh->GetTotalTriangleCount());
      for(uint j = 0; j < mesh->GetTotalTriangleCount(); ++j)
        tmp_triangles[offset + j] = Triangle(mtris[j]);
    }
  }

  this->vertex_count     = tmp_vertexes.size();
  this->normals_count     = tmp_normals.size();
  this->colors_count     = tmp_colors.size();
  this->uvs_count        = tmp_uvs.size();
  this->triangles_count  = tmp_triangles.size();
  this->mesh_descs_count = tmp_mesh_descs.size();

  reset_array(this->vertexes,   this->vertex_count);
  reset_array(this->normals,    this->normals_count);
  reset_array(this->colors,     this->colors_count);
  reset_array(this->uvs,        this->uvs_count);
  reset_array(this->triangles,  this->triangles_count);
  reset_array(this->mesh_descs, this->mesh_descs_count);

  memcpy(vertexes,   &tmp_vertexes[0],   sizeof(Point)    * this->vertex_count);
  memcpy(normals,    &tmp_normals[0],    sizeof(Normal)   * this->normals_count);
  memcpy(colors,     &tmp_colors[0],     sizeof(Spectrum) * this->colors_count);
  memcpy(uvs,        &tmp_uvs[0],        sizeof(UV)       * this->uvs_count);
  memcpy(triangles,  &tmp_triangles[0],  sizeof(Triangle) * this->triangles_count);
  memcpy(mesh_descs, &tmp_mesh_descs[0], sizeof(Mesh)     * this->mesh_descs_count);
}

void PtrFreeScene :: translate_geometry() {
  //mesh_first_triangle_offset.resize(0);
  delete_array(mesh_first_triangle_offset);
  const uint n_vertices  = data_set->GetTotalVertexCount();
  const uint n_triangles = data_set->GetTotalTriangleCount();

  // translate mesh normals, colors and uvs
  this->vertex_count = n_vertices;
  this->normals_count = n_vertices;
  this->colors_count = n_vertices;
  this->uvs_count = n_vertices;
  this->triangles_count = n_triangles;

  reset_array(this->vertexes, this->vertex_count);
  reset_array(this->normals, this->normals_count);
  reset_array(this->colors, this->colors_count);
  reset_array(this->uvs, this->uvs_count);
  reset_array(this->triangles, this->triangles_count);
  //normals.resize(n_vertices);
  //colors.resize(n_vertices);
  //uvs.resize(n_vertices);
  uint index = 0;

  // aux data to later translate triangles
  uint *mesh_offsets = new uint[original_scene->objects.size()];
  uint v_index = 0;

  for (uint i = 0; i < original_scene->objects.size(); ++i) {
    const luxrays::ExtMesh* mesh = original_scene->objects[i];

    mesh_offsets[i] = v_index;
    for(uint j = 0; j < mesh->GetTotalVertexCount(); ++j) {
      normals[index]  = Normal(mesh->GetNormal(j));
      colors[index]   = Spectrum(mesh->GetColor(j));
      uvs[index]      = (mesh->HasUVs()) ? mesh->GetUV(j) : UV(0.f, 0.f);
      vertexes[index] = Point(mesh->GetVertex(j));
      index++;
    }
    v_index += mesh->GetTotalVertexCount();
  }

  // translate mesh triangles
  //triangles.resize(n_triangles);
  index = 0;
  for(uint i = 0; i < original_scene->objects.size(); ++i) {
    const luxrays::ExtMesh* mesh   = original_scene->objects[i];
    const luxrays::Triangle *mtris = mesh->GetTriangles();
    const uint moffset = mesh_offsets[i];
    for (uint j = 0; j < mesh->GetTotalTriangleCount(); ++j) {
      triangles[index++] = Triangle(
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

  // Vertexes checked
  os << "Vertexes (" << scene.vertex_count << "):\n\t";
  for(uint i(0); i < scene.vertex_count; ++i)
    os << scene.vertexes[i] << "\n\t";

  // Normals checked
  os <<  "\n\nNormals (" << scene.normals_count << "):\n\t";
  for(uint i(0); i < scene.normals_count; ++i)
    os << scene.normals[i] << '\n';

  // Colors checked
  os <<  "\n\nColors (" << scene.colors_count << "):\n\t";
  for(uint i(0); i < scene.colors_count; ++i)
    os << scene.colors[i] << "\n\t";

  // UVs checked
  os << "\n\nUVs:\n\t";
  for(uint i(0); i < scene.uvs_count; ++i)
    os << scene.uvs[i] << "\n\t";

  // Triangles checked
  os << "\n\nTriangles (" << scene.triangles_count << "):\n\t";
  for(uint i(0); i < scene.triangles_count; ++i)
    os << scene.triangles[i] << "\n\t";

  // MeshDescs checked
  os << "\n\nMeshDescs:\n\t";
  for(uint i(0); i < scene.mesh_descs_count; ++i)
    os << scene.mesh_descs[i] << "\n\t";

  // MeshIDs checked
  os << "\n\nMeshIDs (" << scene.triangles_count << "):\n\t";
  for(uint i(0); i < scene.triangles_count; ++i)
    os << scene.mesh_ids[i] << "\n\t";

  // MeshFirstTriangleOffset checked
  os << "\n\nMeshFirstTriangleOffset:\n\t";
  for(uint i(0); i < scene.mesh_count; ++i)
    os << scene.mesh_first_triangle_offset[i] << "\n\t";


  // BSphere checked
  os << "\n\nBSphere:\n\t" << scene.bsphere << "\n\t";

  // Camera checked
  os << "\n\nCamera:\n\t" << scene.camera << "\n\t";
//

  // Compiled Materials checked
  os << "\n\nCompiledMaterials:\n\t";
  for(uint i(0); i < scene.compiled_materials_count; ++i)
    os << scene.compiled_materials[i] << "\n\t";

  // Materials checked
  os << "\n\nMaterials:\n\t";
  for(uint i(0); i < scene.materials_count; ++i)
    os << scene.materials[i] << "\n\t";

  // MeshMaterials checked
  os << "\n\nMeshMaterials:\n\t";
  for(uint i(0); i < scene.mesh_materials_count; ++i)
    os << scene.mesh_materials[i] << "\n\t";

  // AreaLights checked
  os << "\n\nAreaLights:\n\t";
  for(uint i(0); i < scene.area_lights_count; ++i)
    os << scene.area_lights[i] << "\n\t";

  // TODO cant check this because there are no values in current mesh
  os << "\n\nInfiniteLight:\n\t" << scene.infinite_light << "\n\t";
  os << "\n\nSunLight:\n\t" << scene.sun_light << "\n\t";
  os << "\n\nSkyLight:\n\t" << scene.sky_light << "\n\t";

  // TODO No TexMaps to check
  //os << "\n\nTexMaps:\n\t";
  //for(uint i(0); i < scene.tex_maps_count; ++i)
  //  os << scene.tex_maps[i] << "\n\t";

  // TODO No RGBTex to check
  //os << "\n\nRGBTex:\n\t";
  //for(uint i(0); i < scene.rgb_tex_count; ++i)
  //  os << scene.rgb_tex[i] << "\n\t";

  // TODO No AlphaTex to check
  //os << "\n\nAlphaTex:\n\t";
  //for(uint i(0); i < scene.alpha_tex_count; ++i)
  //  os << scene.alpha_tex[i] << "\n\t";

  // TODO Can't check. No MeshTexs to check
//  os << "\n\nMeshTexs:\n\t";
//  for(uint i(0); i < scene.mesh_texs_count; ++i)
//    os << scene.mesh_texs[i] << "\n\t";

  // TODO Can't check. No BumpMap to check
//  os << "\n\nBumpMap:\n\t";
//  for(uint i(0); i < scene.bump_map_count; ++i)
//    os << scene.bump_map[i] << "\n\t";

  // TODO Can't check. No BumpMapScales to check
//  os << "\n\nBumpMapScales:\n\t";
//  for(uint i(0); i < scene.bump_map_scales_count; ++i)
//    os << scene.bump_map_scales[i] << "\n\t";

  // TODO Can't check. No NormalMap to check
//  os << "\n\nNormalMap:\n\t";
//  for(uint i(0); i < scene.normal_map_count; ++i)
//    os << scene.normal_map[i] << "\n\t";

  return os;
}

PtrFreeScene* PtrFreeScene :: to_device(int device_id) const {
  cudaSetDevice(device_id);

  PtrFreeScene* scene = new PtrFreeScene;
  memcpy(scene, this, sizeof(PtrFreeScene));
  scene->original_scene = NULL;
  scene->data_set = NULL;

  /*Point* vertexes;
  cudaMalloc(&vertexes, sizeof(Point) * vertex_count);
  cudaMemcpy(vertexes, this->vertexes, sizeof(Point) * vertex_count, cudaMemcpyHostToDevice);

  test_kernel<<<1, 1>>>(vertexes, vertex_count);
  Point new_vertexes[2];
  cudaMemcpy(&new_vertexes, vertexes, 2 * sizeof(Point), cudaMemcpyDeviceToHost);
  printf("%f\n", new_vertexes[0].x);*/

  CUDA_SAFE(alloc_copy_to_cuda(&scene->vertexes,   this->vertexes,   vertex_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->normals,    this->normals,    normals_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->colors,     this->colors,     colors_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->uvs,        this->uvs,        uvs_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->triangles,  this->triangles,  triangles_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->mesh_descs, this->mesh_descs, mesh_descs_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->mesh_ids,   this->mesh_ids,   data_set->totalTriangleCount));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->mesh_first_triangle_offset, this->mesh_first_triangle_offset, mesh_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->compiled_materials, this->compiled_materials, ppm::MAT_MAX));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->materials,          this->materials,          materials_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->mesh_materials,     this->mesh_materials,     mesh_materials_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->area_lights,        this->area_lights,        area_lights_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->infinite_light_map, this->infinite_light_map, this->infinite_light.width * this->infinite_light.height));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->tex_maps,           this->tex_maps,         tex_maps_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->rgb_tex,            this->rgb_tex,          rgb_tex_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->alpha_tex,          this->alpha_tex,        alpha_tex_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->mesh_texs,          this->mesh_texs,        mesh_materials_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->bump_map,           this->bump_map,         mesh_materials_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->bump_map_scales,    this->bump_map_scales,  mesh_materials_count));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->normal_map,         this->normal_map,       mesh_materials_count));

  CUDA_SAFE(alloc_copy_to_cuda(&scene->nodes, (QBVHNode*)    this->data_set->GetAccelerator()->GetNodes(), n_nodes));
  CUDA_SAFE(alloc_copy_to_cuda(&scene->prims, (QuadTriangle*)this->data_set->GetAccelerator()->GetPrims(), n_prims));

  PtrFreeScene* cuda_scene;
  CUDA_SAFE(alloc_copy_to_cuda(&cuda_scene, scene, 1));
  CUDA_SAFE(cudaDeviceSynchronize());

  return cuda_scene;
}

}
