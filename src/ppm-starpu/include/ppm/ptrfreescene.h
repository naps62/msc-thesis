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
  typedef bool(*lux_mesh_comparator_t)(luxrays::Mesh*, luxrays::Mesh*);
  typedef std::map<luxrays::ExtMesh*, uint, lux_mesh_comparator_t> lux_defined_meshs_t;

private:
  const Config& config;           // reference to global configs
  luxrays::Scene* original_scene; // original scene structure
  luxrays::DataSet* data_set;     // original data_set structure

  std::vector<Point>    vertexes;
  std::vector<Normal>   normals;
  std::vector<Spectrum> colors;
  std::vector<UV>       uvs;
  std::vector<Triangle> triangles;
  std::vector<Mesh>     mesh_descs;

  std::vector<uint> mesh_ids;
  //const uint* mesh_ids;

  std::vector<uint> mesh_first_triangle_offset;
  BSphere bsphere; // bounding sphere of the scene
  Camera camera;   // compiled camera

  // materials
  std::vector<bool>     compiled_materials;
  std::vector<Material> materials;
  std::vector<uint>     mesh_mats;

  // lights
  std::vector<TriangleLight> area_lights;
  InfiniteLight  infinite_light;
  SunLight       sun_light;
  SkyLight       sky_light;

  // textures
  std::vector<TexMap> tex_maps;
  std::vector<Spectrum> rgb_tex;
  std::vector<float> alpha_tex;
  std::vector<uint> mesh_texs;

  // bump maps
  std::vector<uint> bump_map;
  std::vector<float> bump_map_scales;

  // normal maps
  std::vector<uint> normal_map;


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
