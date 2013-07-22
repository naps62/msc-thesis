#include "ppm/kernels/helpers.cuh"
#include "ppm/ptrfree_hash_grid.h"
#include "ppm/math.h"
#include <limits>
#include <cfloat>

namespace ppm { namespace kernels {

namespace helpers {

__HD__
void concentric_sample_disk(const float u1, const float u2, float *dx, float *dy) {
  float r, theta;
  // Map uniform random numbers to $[-1,1]^2$
  float sx = 2.f * u1 - 1.f;
  float sy = 2.f * u2 - 1.f;
  // Map square to $(r,\theta)$
  // Handle degeneracy at the origin
  if (sx == 0.f && sy == 0.f) {
    *dx = 0.f;
    *dy = 0.f;
    return;
  }
  if (sx >= -sy) {
    if (sx > sy) {
      // Handle first region of disk
      r = sx;
      if (sy > 0.f)
        theta = sy / r;
      else
        theta = 8.f + sy / r;
    } else {
      // Handle second region of disk
      r = sy;
      theta = 2.f - sx / r;
    }
  } else {
    if (sx <= sy) {
      // Handle third region of disk
      r = -sx;
      theta = 4.f - sy / r;
    } else {
      // Handle fourth region of disk
      r = -sy;
      theta = 6.f + sx / r;
    }
  }
  theta *= M_PI / 4.f;
  *dx = r * cosf(theta);
  *dy = r * sinf(theta);
}

__HD__
Ray generate_ray(
    const float sx, const float sy,
    const uint width, const uint height,
    const float u0, const float u1, const float u2, const Camera& camera) {

  Point p(sx, height - sy - 1.f, 0);
  Point orig;

  const float iw = 1.f / (camera.raster_to_camera_matrix[3][0] * p.x
                        + camera.raster_to_camera_matrix[3][1] * p.y
                        + camera.raster_to_camera_matrix[3][2] * p.z
                        + camera.raster_to_camera_matrix[3][3]);
  orig.x = (camera.raster_to_camera_matrix[0][0] * p.x
      + camera.raster_to_camera_matrix[0][1] * p.y
      + camera.raster_to_camera_matrix[0][2] * p.z
      + camera.raster_to_camera_matrix[0][3]) * iw;
  orig.y = (camera.raster_to_camera_matrix[1][0] * p.x
      + camera.raster_to_camera_matrix[1][1] * p.y
      + camera.raster_to_camera_matrix[1][2] * p.z
      + camera.raster_to_camera_matrix[1][3]) * iw;
  orig.z = (camera.raster_to_camera_matrix[2][0] * p.x
      + camera.raster_to_camera_matrix[2][1] * p.y
      + camera.raster_to_camera_matrix[2][2] * p.z
      + camera.raster_to_camera_matrix[2][3]) * iw;

  Vector dir(orig);

  const float hither = camera.hither;
  if (camera.lens_radius > 0.f) {
    // sample point on lens
    float lens_u, lens_v;
    concentric_sample_disk(u1, u2, &lens_u, &lens_v);
    const float lens_radius = camera.lens_radius;
    lens_u *= lens_radius;
    lens_v *= lens_radius;

    // compute point on plane of focus
    const float focal_distance = camera.focal_distance;
    const float dist = focal_distance - hither;
    const float ft = dist / dir.z;
    Point p_focus = orig + dir * ft;

    // update ray for effect on lens
    const float k = dist / focal_distance;
    orig.x += lens_u * k;
    orig.y += lens_v * k;

    dir = p_focus - orig;
  }

  dir = Normalize(dir);

  Point torig;
  const float iw2 = 1.f / ( camera.camera_to_world_matrix[3][0] * orig.x
                      + camera.camera_to_world_matrix[3][1] * orig.y
                      + camera.camera_to_world_matrix[3][2] * orig.z
                      + camera.camera_to_world_matrix[3][3]);
  torig.x = (camera.camera_to_world_matrix[0][0] * orig.x
      +  camera.camera_to_world_matrix[0][1] * orig.y
      +  camera.camera_to_world_matrix[0][2] * orig.z
      +  camera.camera_to_world_matrix[0][3]) * iw2;
  torig.y = (camera.camera_to_world_matrix[1][0] * orig.x
      +  camera.camera_to_world_matrix[1][1] * orig.y
      +  camera.camera_to_world_matrix[1][2] * orig.z
      +  camera.camera_to_world_matrix[1][3]) * iw2;
  torig.z = (camera.camera_to_world_matrix[2][0] * orig.x
      +  camera.camera_to_world_matrix[2][1] * orig.y
      +  camera.camera_to_world_matrix[2][2] * orig.z
      +  camera.camera_to_world_matrix[2][3]) * iw2;

  Vector tdir;
  tdir.x = camera.camera_to_world_matrix[0][0] * dir.x
       + camera.camera_to_world_matrix[0][1] * dir.y
       + camera.camera_to_world_matrix[0][2] * dir.z;
  tdir.y = camera.camera_to_world_matrix[1][0] * dir.x
       + camera.camera_to_world_matrix[1][1] * dir.y
       + camera.camera_to_world_matrix[1][2] * dir.z;
  tdir.z = camera.camera_to_world_matrix[2][0] * dir.x
       + camera.camera_to_world_matrix[2][1] * dir.y
       + camera.camera_to_world_matrix[2][2] * dir.z;

  return Ray(torig, tdir, RAY_EPSILON, (camera.yon - hither) / dir.z);
}

__HD__
void tex_map_get_texel(
    const Spectrum* const pixels,
    const unsigned width,
    const unsigned height,
    const int s,
    const int t,
    Spectrum& color) {

  const unsigned u = Mod(s, width);
  const unsigned v = Mod(s, height);

  const Spectrum& pixel = pixels[v * width + u];

  color.r = pixel.r;
  color.g = pixel.g;
  color.b = pixel.b;
}

__HD__
void tex_map_get_color(
    const Spectrum* const pixels,
    const unsigned width,
    const unsigned height,
    const float u,
    const float v,
    Spectrum& color) {

  const float s = u * width  - 0.5f;
  const float t = v * height - 0.5f;

  const int s0 = (int) floor(s);
  const int t0 = (int) floor(t);

  const float ds = s - s0;
  const float dt = t - t0;

  const float ids = 1.f - ds;
  const float idt = 1.f - dt;

  Spectrum c0, c1, c2, c3;
  tex_map_get_texel(pixels, width, height, s0,     t0,     c0);
  tex_map_get_texel(pixels, width, height, s0,     t0 + 1, c1);
  tex_map_get_texel(pixels, width, height, s0 + 1, t0,     c2);
  tex_map_get_texel(pixels, width, height, s0 + 1, t0 + 1, c3);

  const float k0 = ids * idt;
  const float k1 = ids * dt;
  const float k2 = ds  * idt;
  const float k3 = ds  * dt;

  color.r = k0 * c0.r + k1 * c1.r + k2 * c2.r + k3 * c3.r;
  color.g = k0 * c0.g + k1 * c1.g + k2 * c2.g + k3 * c3.g;
  color.b = k0 * c0.b + k1 * c1.b + k2 * c2.b + k3 * c3.b;
}


__HD__
void infinite_light_le (
    Spectrum& le,
    const Vector& dir,
    const InfiniteLight& infinite_light,
    const Spectrum* const infinite_light_map) {

  const float u = 1.f - SphericalPhi(dir)   * INV_TWOPI + infinite_light.shiftU;
  const float v =       SphericalTheta(dir) * INV_PI    + infinite_light.shiftV;

  tex_map_get_color(infinite_light_map, infinite_light.width, infinite_light.height, u, v, le);

  //std::cout << le << '\n';
  le.r *= infinite_light.gain.r;
  le.g *= infinite_light.gain.g;
  le.b *= infinite_light.gain.b;

}


__HD__
void sky_light_le(
    Spectrum& f,
    const Vector& dir,
    const SkyLight& sky_light) {

  const float theta = SphericalTheta(dir);
  const float phi   = SphericalPhi(dir);

  Spectrum s;
  sky_light_get_sky_spectral_radiance(theta, phi, s, sky_light);

  f.r = sky_light.gain.r * s.r;
  f.g = sky_light.gain.g * s.g;
  f.b = sky_light.gain.b * s.b;
}

__HD__
float sky_light_perez_base(
    const float* const lam,
    const float theta,
    const float gamma) {

  return (1.f + lam[1] * exp(lam[2] / cos(theta))) * (1.f + lam[3] * exp(lam[4] * gamma) + lam[5] * cos(gamma) * cos(gamma));
}

__HD__
void sky_light_chromaticity_to_spectrum(
    const float Y,
    const float x,
    const float y,
    Spectrum& s) {

  float X, Z;

  if (y != 0.f)
    X = (x/y) * Y;
  else
    X = 0.f;

  if (y != 0.f && Y != 0.f)
    Z = (1.f - x - y) / y * Y;
  else
    Z = 0.f;

  // assuming sRGB (D65 illuminant)
  s.r =  3.2410f * X - 1.5374f * Y - 0.4986f * Z;
  s.g = -0.9692f * X + 1.8760f * Y + 0.0416f * Z;
  s.b =  0.0556f * X - 0.2040f * Y + 1.0570f * Z;
}

__HD__
float ri_angle_between(
    const float thetav,
    const float phiv,
    const float theta,
    const float phi) {

  const float cospsi = sinf(thetav) * sin(theta) * cosf(phi - phiv) + cosf(thetav) * cosf(theta);

  if (cospsi >= 1.f)
    return 0.f;
  if (cospsi <= -1.f)
    return M_PI;
  return acosf(cospsi);
}

__HD__
void sky_light_get_sky_spectral_radiance(
    const float theta,
    const float phi,
    Spectrum& spect,
    const SkyLight& sky_light) {

  const float theta_fin = min(theta, (float) ((M_PI * 0.5f) - 0.001f));
  const float gamma     = ri_angle_between(theta, phi, sky_light.theta_s, sky_light.phi_s);

  const float x = sky_light.zenith_x * sky_light_perez_base(sky_light.perez_x, theta_fin, gamma);
  const float y = sky_light.zenith_y * sky_light_perez_base(sky_light.perez_y, theta_fin, gamma);
  const float Y = sky_light.zenith_Y * sky_light_perez_base(sky_light.perez_Y, theta_fin, gamma);

  sky_light_chromaticity_to_spectrum(Y, x, y, spect);
}


__HD__
void sun_light_le(
    Spectrum& le,
    const Vector& dir,
    const SunLight& sun_light) {

  const float cos_theta_max = sun_light.cos_theta_max;
  const Vector sun_dir = sun_light.dir;

  if ((cos_theta_max < 1.f) && (Dot(dir, sun_dir) > cos_theta_max))
    le = sun_light.color;
  else {
    le.r = 0.f;
    le.g = 0.f;
    le.b = 0.f;
  }
}

__HD__
void area_light_le(
    Spectrum& le,
    const Vector& wo,
    const Normal& light_normal,
    const AreaLightParam& mat) {

  const bool bright_side = (Dot(light_normal, wo) > 0.f);

  if (bright_side) {
    le.r = mat.gain.r;
    le.g = mat.gain.g;
    le.b = mat.gain.b;
  } else {
    le.r = 0.f;
    le.g = 0.f;
    le.b = 0.f;
  }
}

__HD__
bool get_hit_point_information(
    const PtrFreeScene* const scene,
    Ray& ray,
    const RayHit& hit,
    Point& hit_point,
    Spectrum& surface_color,
    Normal& N,
    Normal& shade_N) {

  hit_point = ray(hit.t);
  const unsigned current_triangle_index = hit.index;

  unsigned current_mesh_index = scene->mesh_ids[current_triangle_index];
  unsigned triangle_index = current_triangle_index - scene->mesh_first_triangle_offset[current_mesh_index];

  const Mesh& m = scene->mesh_descs[current_mesh_index];

  if (m.has_colors) {
    // mesh interpolate color
    mesh_interpolate_color(&scene->colors[m.colors_offset], &scene->triangles[m.tris_offset], triangle_index, hit.b1, hit.b2, surface_color);
  } else {
    surface_color = Spectrum(1.f, 1.f, 1.f);
  }

  // mesh interpolate normal
  mesh_interpolate_normal(&scene->normals[m.verts_offset], &scene->triangles[m.tris_offset], triangle_index, hit.b1, hit.b2, N);

  if (Dot(ray.d, N) > 0.f)
    shade_N = -N;
  else
    shade_N = N;

  return false;

}

__HD__
void mesh_interpolate_color(
    const Spectrum* const colors,
    const Triangle* const triangles,
    const unsigned triangle_index,
    const float b1,
    const float b2,
    Spectrum& C) {

  const Triangle& triangle = triangles[triangle_index];
  const float b0 = 1.f - b1 - b2;

  C.r = b0 * colors[triangle.v[0]].r + b1 * colors[triangle.v[1]].r + b2 * colors[triangle.v[2]].r;
  C.g = b0 * colors[triangle.v[0]].g + b1 * colors[triangle.v[1]].g + b2 * colors[triangle.v[2]].g;
  C.b = b0 * colors[triangle.v[0]].b + b1 * colors[triangle.v[1]].b + b2 * colors[triangle.v[2]].b;
}

__HD__
void mesh_interpolate_normal(
    const Normal* const normals,
    const Triangle* const triangles,
    const unsigned triangle_index,
    const float b1,
    const float b2,
    Normal& N) {

  const Triangle& triangle = triangles[triangle_index];
  const float b0 = 1.f - b1 - b2;

  const Normal& v0 = normals[triangle.v[0]];
  const Normal& v1 = normals[triangle.v[1]];
  const Normal& v2 = normals[triangle.v[2]];

  N.x = b0 * v0.x + b1 * v1.x + b2 * v2.x;
  N.y = b0 * v0.y + b1 * v1.y + b2 * v2.y;
  N.z = b0 * v0.z + b1 * v1.z + b2 * v2.z;

  N = Normalize(N);
}

__HD__
void mesh_interpolate_UV(
    const UV* const uvs,
    const Triangle* const triangles,
    const unsigned triangle_index,
    const float b1,
    const float b2,
    UV& uv) {

  const Triangle& triangle = triangles[triangle_index];
  const float b0 = 1.f - b1 - b2;

  uv.u = b0 * uvs[triangle.v[0]].u + b1 * uvs[triangle.v[1]].u + b2 * uvs[triangle.v[2]].u;
  uv.v = b0 * uvs[triangle.v[0]].v + b1 * uvs[triangle.v[1]].v + b2 * uvs[triangle.v[2]].v;
}



__HD__
LightType sample_all_lights(
    const float u,
    const unsigned lights_count,
    const InfiniteLight& infinite_light,
    const SunLight& sun_light,
    const SkyLight& sky_light,
    float& pdf,
    int& light_index,
    const bool skip_infinite_light) {

  if (!skip_infinite_light && (infinite_light.exists || sun_light.exists || sky_light.exists)) {
    unsigned count = lights_count;
    int ilx1 = 0;
    int ilx2 = 0;
    int ilx3 = 0;

    if (infinite_light.exists) ilx1 = count++;
    if (sun_light.exists)      ilx2 = count++;
    if (sky_light.exists)      ilx3 = count++;

    light_index = Floor2UInt(count * u);
    pdf = 1.f / count;

    if      (light_index == ilx1) return ppm::LIGHT_IL_IS;
    else if (light_index == ilx2) return ppm::LIGHT_SUN;
    else if (light_index == ilx3) return ppm::LIGHT_IL_SKY;
    else return ppm::LIGHT_TRIANGLE;

  } else {
    light_index = Min(Floor2UInt(lights_count * u), lights_count - 1);
    pdf = 1.f / lights_count;
    return ppm::LIGHT_TRIANGLE;
  }
}

__HD__
void infinite_light_sample_l(
    const float u0,
    const float u1,
    const float u2,
    const float u3,
    const InfiniteLight& infinite_light,
    const Spectrum* const infinite_light_map,
    const BSphere& bsphere,
    float& pdf, Ray& ray, Spectrum& f) {

  const float rad = bsphere.rad * 1.01f;
  const Point p1 = bsphere.center + rad * UniformSampleSphere(u0, u1);
  const Point p2 = bsphere.center + rad * UniformSampleSphere(u2, u3);

  ray = Ray(p1, Normalize(p2 - p1));

  const Vector to_center = Normalize(bsphere.center - p1);
  const float cos_theta = AbsDot(to_center, ray.d);
  pdf = cos_theta / (4.f * M_PI * M_PI * rad * rad);

  Vector dir = -ray.d;
  infinite_light_le(f, dir, infinite_light, infinite_light_map);
}

__HD__
void sun_light_sample_l(
    const float u0,
    const float u1, const SunLight& sun_light,
    const Point& hit_point,
    float& pdf,
    Ray& shadow_ray,
    Spectrum& f) {

  Vector wi = UniformSampleCone(u0, u1, sun_light.cos_theta_max, sun_light.x, sun_light.y, sun_light.dir);

  shadow_ray.o = hit_point;
  shadow_ray.d = wi;
  shadow_ray.mint = RAY_EPSILON;
  shadow_ray.maxt = FLT_MAX/*std::numeric_limits<float>::max()*/;

  f = sun_light.color;
  pdf = UniformConePdf(sun_light.cos_theta_max);
}

__HD__
void sun_light_sample_l(
    const float u0,
    const float u1,
    const float u2,
    const float u3,
    const SunLight& sun_light,
    const BSphere& bsphere,
    float& pdf,
    Ray& ray,
    Spectrum& f) {

  const float rad = bsphere.rad * 1.01f;

  float d1, d2;
  ConcentricSampleDisk(u0, u1, &d1, &d2);
  const Point p_disk = bsphere.center + rad * (d1 * sun_light.x + d2 * sun_light.y);

  ray = Ray(p_disk + rad * sun_light.dir, -UniformSampleCone(u2, u3, sun_light.cos_theta_max, sun_light.x, sun_light.y, sun_light.dir));
  pdf = UniformConePdf(sun_light.cos_theta_max) / (M_PI * rad * rad);
}

__HD__
void sky_light_sample_l(
    const float u0,
    const float u1,
    const float u2,
    const float u3,
    const SkyLight& sky_light,
    const BSphere& bsphere,
    float& pdf,
    Ray& ray,
    Spectrum& f) {

  const float rad = bsphere.rad * 1.01f;
  const Point p1 = bsphere.center + rad * UniformSampleSphere(u0, u1);
  const Point p2 = bsphere.center + rad * UniformSampleSphere(u2, u3);

  ray = Ray(p1, Normalize(p2 - p1));

  const Vector to_center = Normalize(bsphere.center - p1);
  const float cos_theta = AbsDot(to_center, ray.d);
  pdf = cos_theta / (4.f * M_PI * M_PI * rad * rad);

  const Vector dir = -ray.d;
  sky_light_le(f, dir, sky_light);
}


__HD__
void triangle_light_sample_l(
    const float u0,
    const float u1,
    const float u2,
    const float u3,
    const TriangleLight& light,
    const Mesh* const mesh_descs,
    const Spectrum* const colors,
    float& pdf,
    Ray& ray,
    Spectrum& f) {

  Point orig;
  sample_triangle_light(light, u0, u1, orig);

  const Normal sample_N = light.normal;

  const float z = 1.f - 2.f * u2;
  const float r = sqrtf(Max(0.f, 1.f - z * z));
  const float phi = 2.f * M_PI * u3;
  const float x = r * cosf(phi);
  const float y = r * sinf(phi);

  Vector dir = Vector(x, y, z);
  float RdotN = Dot(dir, sample_N);
  if (RdotN < 0.f) {
    dir *= -1.f;
    RdotN = -RdotN;
  }

  ray = Ray(orig, dir);
  pdf = INV_TWOPI / light.area;
  const Mesh& m = mesh_descs[light.mesh_index];

  f.r = light.gain.r * RdotN;
  f.g = light.gain.g * RdotN;
  f.b = light.gain.b * RdotN;

  if (m.has_colors) {
    const unsigned i = m.colors_offset + light.tri_index;
    f.r *= colors[i].r;
    f.g *= colors[i].g;
    f.b *= colors[i].b;
  }
}

__HD__
void sample_triangle_light(
    const TriangleLight& l,
    const float u0,
    const float u1,
    Point& p) {

  const float su1 = sqrt(u0);
  const float b0 = 1.f - su1;
  const float b1 = u1 * su1;
  const float b2 = 1.f - b0 - b1;

  p.x = b0 * l.v0.x + b1 * l.v1.x + b2 * l.v2.x;
  p.y = b0 * l.v0.y + b1 * l.v1.y + b2 * l.v2.y;
  p.z = b0 * l.v0.z + b1 * l.v1.z + b2 * l.v2.z;
}


}

} }
