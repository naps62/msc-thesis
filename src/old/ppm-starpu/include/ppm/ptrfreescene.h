#ifndef _PPM_PTRFREESCENE_H_
#define _PPM_PTRFREESCENE_H_

#include "utils/common.h"
#include "utils/config.h"
#include "ppm/types.h"
#include "luxrays/utils/sdl/scene.h"
#include "luxrays/core/dataset.h"
#include "utils/action_list.h"

namespace ppm {

class PtrFreeScene {

public:
  PtrFreeScene(const Config& config);
  ~PtrFreeScene();

  void recompile(const ActionList& actions);

  Ray generate_ray(const float sx, const float sy, const uint width, const uint height, const float u0, const float u1, const float u2) const;

  bool intersect(Ray& ray, RayHit& hit) const;

  typedef std::vector<luxrays::ExtMesh*> lux_ext_mesh_list_t;
  //typedef luxrays::ExtMesh* lux_ext_mesh_list_t[];

  typedef bool(*lux_mesh_comparator_t)(luxrays::Mesh*, luxrays::Mesh*);
  typedef std::map<luxrays::ExtMesh*, uint, lux_mesh_comparator_t> lux_defined_meshs_t;
  struct lux_defined_meshs_pair_t {
    luxrays::ExtMesh* mesh;
    uint i;
    lux_mesh_comparator_t comparator;
  };
  typedef lux_defined_meshs_pair_t lux_defined_meshs_array_t[];

//private:
  const Config& config;           // reference to global configs
  luxrays::Scene* original_scene; // original scene structure
  luxrays::DataSet* data_set;     // original data_set structure

  unsigned mesh_count;
  unsigned compiled_materials_count;
  unsigned materials_count;
  unsigned mesh_materials_count;
  unsigned area_lights_count;
  unsigned rgb_tex_count;
  unsigned alpha_tex_count;
  unsigned tex_maps_count;

  unsigned vertex_count;
  unsigned normals_count;
  unsigned colors_count;
  unsigned uvs_count;
  unsigned triangles_count;
  unsigned mesh_descs_count;
  Point*    vertexes;
  Normal*   normals;
  Spectrum* colors;
  UV*       uvs;
  Triangle* triangles;
  Mesh*     mesh_descs;

  uint* mesh_ids; // size == mesh_count

  uint* mesh_first_triangle_offset;
  BSphere bsphere; // bounding sphere of the scene
  Camera camera;   // compiled camera

  // materials
  bool*     compiled_materials; // size = ppm::MAT_MAX
  Material* materials;
  uint*     mesh_materials; // size = mesh_mats_count

  // lights
  TriangleLight*  area_lights;
  InfiniteLight   infinite_light;
  const Spectrum* infinite_light_map;
  SunLight        sun_light;
  SkyLight        sky_light;

  // textures
  TexMap*   tex_maps;
  Spectrum* rgb_tex;
  float*    alpha_tex;
  uint*     mesh_texs;

  // bump maps
  uint*  bump_map;
  float* bump_map_scales;

  // normal maps
  uint* normal_map;


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
//  static lux_mesh_comparator_t mesh_ptr_compare;
  static bool mesh_ptr_compare(luxrays::Mesh* m0, luxrays::Mesh* m1);

};

}

#endif // _PPM_PTRFREESCENE_H_
