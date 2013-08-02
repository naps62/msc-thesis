#ifndef _PPM_KERNELS_HELPERS_H_
#define _PPM_KERNELS_HELPERS_H_

#include "ppm/types.h"
#include "ppm/ptrfreescene.h"
#include "ppm/ptrfree_hash_grid.h"
using namespace ppm;

namespace ppm { namespace kernels {

namespace helpers {


__HD__ void concentric_sample_disk(const float u1, const float u2, float *dx, float *dy);
__HD__ Ray generate_ray(const float sx, const float sy, const uint width, const uint height, const float u0, const float u1, const float u2, const Camera& camera);

__HD__ void tex_map_get_texel(const Spectrum* const pixels, const unsigned width, const unsigned height, const int s, const int t, Spectrum& color);
__HD__ void tex_map_get_color(const Spectrum* const pixels, const unsigned width, const unsigned height, const float u, const float v, Spectrum& color);

__HD__ void infinite_light_le (Spectrum& le, const Vector& dir, const InfiniteLight& infinite_light, const Spectrum* const infinite_light_map);
__HD__ void sky_light_le      (Spectrum& le, const Vector& dir, const SkyLight& sky_light);
__HD__ void sun_light_le      (Spectrum& le, const Vector& dir, const SunLight& sun_light);
__HD__ void area_light_le     (Spectrum& le, const Vector& wo, const Normal& light_normal, const AreaLightParam& mat);

__HD__ float sky_light_perez_base(const float* const lam, const float theta, const float gamma);
__HD__ void sky_light_chromaticity_to_spectrum(const float Y, const float x, const float y, Spectrum& s);
__HD__ float ri_angle_between(const float thetav, const float phiv, const float theta, const float phi);
__HD__ void sky_light_get_sky_spectral_radiance(const float theta, const float phi, Spectrum& spect, const SkyLight& sky_light);

__HD__ bool get_hit_point_information(const PtrFreeScene* const scene, Ray& ray, const RayHit& hit, Point& hit_point, Spectrum& surface_color, Normal& N, Normal& shade_N);

__HD__ void mesh_interpolate_color (const Spectrum* const colors, const Triangle* const triangles, const unsigned triangle_index, const float b1, const float b2, Spectrum& C);
__HD__ void mesh_interpolate_normal(const Normal* const normals,  const Triangle* const triangles, const unsigned triangle_index, const float b1, const float b2, Normal& N);
__HD__ void mesh_interpolate_UV    (const UV* const uvs,          const Triangle* const triangles, const unsigned triangle_index, const float b1, const float b2, UV& uv);

__HD__ void generic_material_sample_f     (const Material& mat,               Vector& wo, Vector& wi, const Normal& N, const Normal& shade_N, const float u0, const float u1, const float u2, float& pdf, Spectrum& f, bool& specular_bounce);
__HD__ void matte_material_sample_f       (const MatteParam& mat,       const Vector& wo, Vector& wi,                  const Normal& shade_N, const float u0, const float u1,                 float& pdf, Spectrum& f, bool& specular_bounce);
__HD__ void mirror_material_sample_f      (const MirrorParam& mat,      const Vector& wo, Vector& wi,                  const Normal& shade_N,                                                 float& pdf, Spectrum& f, bool& specular_bounce);
__HD__ void glass_material_sample_f       (const GlassParam& mat,       const Vector& wo, Vector& wi, const Normal& N, const Normal& shade_N, const float u0,                                 float& pdf, Spectrum& f, bool& specular_bounce);
__HD__ void matte_mirror_material_sample_f(const MatteMirrorParam& mat, const Vector& wo, Vector& wi,                  const Normal& shade_N, const float u0, const float u1, const float u2, float& pdf, Spectrum& f, bool& specular_bounce);
__HD__ void metal_material_sample_f       (const MetalParam& mat,       const Vector& wo, Vector& wi,                  const Normal& shade_N, const float u0, const float u1,                 float& pdf, Spectrum& f, bool& specular_bounce);
__HD__ void matte_metal_material_sample_f (const MatteMetalParam& mat,  const Vector& w0, Vector& wi,                  const Normal& shade_N, const float u0, const float u1, const float u2, float& pdf, Spectrum& f, bool& specular_bounce);
__HD__ void alloy_material_sample_f       (const AlloyParam& mat,       const Vector& wo, Vector& wi,                  const Normal& shade_N, const float u0, const float u1, const float u2, float& pdf, Spectrum& f, bool& specular_bounce);
__HD__ void arch_glass_material_sample_f  (const ArchGlassParam& mat,   const Vector& wo, Vector& wi, const Normal& N, const Normal& shade_N, const float u0,                                 float& pdf, Spectrum& f, bool& specular_bounce);

__HD__ void glossy_reflection(const Vector& wo, Vector& wi, const float exponent, const Normal& shade_N, const float u0, const float u1);

__HD__ LightType sample_all_lights(const float u, const unsigned lights_count, const InfiniteLight& infinite_light, const SunLight& sun_light, const SkyLight& sky_light, float& pdf, int& light_index, const bool skip_inifinite_ligth = false);
__HD__ void infinite_light_sample_l(const float u0, const float u1, const float u2, const float u3, const InfiniteLight& infinite_light,    const Spectrum* const infinite_light_map, const BSphere& bsphere, float& pdf, Ray& ray, Spectrum& f);
__HD__ void sun_light_sample_l     (const float u0, const float u1,                                 const SunLight& sun_light, const Point& hit_point,                                      float& pdf, Ray& shadow_ray, Spectrum& f);
__HD__ void sun_light_sample_l     (const float u0, const float u1, const float u2, const float u3, const SunLight& sun_light, const BSphere& bsphere,                                      float& pdf, Ray& ray,        Spectrum& f);
__HD__ void sky_light_sample_l     (const float u0, const float u1, const float u2, const float u3, const SkyLight& sky_light, const BSphere& bsphere,                                      float& pdf, Ray& ray,        Spectrum& f);
__HD__ void triangle_light_sample_l(const float u0, const float u1, const float u2, const float u3, const TriangleLight& light, const Mesh* const mesh_descs, const Spectrum* const colors, float& pdf, Ray& ray,        Spectrum& f);

__HD__ void sample_triangle_light(const TriangleLight& l, const float u0, const float u1, Point& p);

__HD__ void matte_f(const MatteParam& mat, Spectrum& f);
__HD__ void matte_mirror_f(const MatteMirrorParam& mat, Spectrum& f);
__HD__ void matte_metal_f(const MatteMetalParam& mat, Spectrum& f);
__HD__ void alloy_f(const MatteParam& mat, const Vector& wo, const Normal& N, Spectrum& f);

//__HD__ void add_flux(const PtrFreeHashGrid* const hash_grid, const BBox& bbox, const PtrFreeScene* const scene, const Point& hit_point, const Normal& shade_N, const Vector& wi, const Spectrum& photon_flux, const float photon_radius2, HitPointPosition* const hit_points_info, HitPointRadiance* const hit_points);
__HD__ void add_flux(
    const unsigned*  hash_grid,
    const unsigned*  hash_grid_lengths,
    const unsigned*  hash_grid_indexes,
    const unsigned   hash_grid_entry_count,
    const float      hash_grid_inv_cell_size,
    const BBox& bbox,
    const PtrFreeScene* const scene,
    const Point& hit_point,
    const Normal& shade_N,
    const Vector& wi,
    const Spectrum& photon_flux,
    const float photon_radius2,
    HitPointPosition* const hit_points_info,
    HitPointRadiance* const hit_points,
    const unsigned hit_points_count);

__HD__ unsigned hash(const int ix, const int iy, const int iz, unsigned size);

}

} }

#endif // _PPM_KERNELS_HELPERS_H_
