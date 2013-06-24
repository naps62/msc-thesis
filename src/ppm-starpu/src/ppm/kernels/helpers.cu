#include "ppm/kernels/helpers.cuh"

namespace ppm { namespace kernels {

namespace helpers {

__HD__
void tex_map_get_texel(const Spectrum* const pixels, const unsigned width, const unsigned height, const int s, const int t, Spectrum& color) {
  const unsigned u = Mod(s, width);
  const unsigned v = Mod(s, height);

  const Spectrum& pixel = pixels[v * width + u];

  color.r = pixel.r;
  color.g = pixel.g;
  color.b = pixel.b;
}

__HD__
void tex_map_get_color(const Spectrum* const pixels, const unsigned width, const unsigned height, const float u, const float v, Spectrum& color) {
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
void infinite_light_le (Spectrum& le, const Vector& dir, const InfiniteLight& infinite_light, const Spectrum* const infinite_light_map) {

  const float u = 1.f - SphericalPhi(dir)   * INV_TWOPI + infinite_light.shiftU;
  const float v =       SphericalTheta(dir) * INV_PI    + infinite_light.shiftV;

  tex_map_get_color(infinite_light_map, infinite_light.width, infinite_light.height, u, v, le);

  le.r *= infinite_light.gain.r;
  le.g *= infinite_light.gain.g;
  le.b *= infinite_light.gain.b;
}


__HD__
void sky_light_le(Spectrum& f, const Vector& dir, const SkyLight& sky_light) {

  const float theta = SphericalTheta(dir);
  const float phi   = SphericalPhi(dir);

  Spectrum s;
  sky_light_get_sky_spectral_radiance(theta, phi, s, sky_light);

  f.r = sky_light.gain.r * s.r;
  f.g = sky_light.gain.g * s.g;
  f.b = sky_light.gain.b * s.b;
}

__HD__
float sky_light_perez_base(const float* const lam, const float theta, const float gamma) {
  return (1.f + lam[1] * exp(lam[2] / cos(theta))) * (1.f + lam[3] * exp(lam[4] * gamma) + lam[5] * cos(gamma) * cos(gamma));
}

__HD__
void sky_light_chromaticity_to_spectrum(const float Y, const float x, const float y, Spectrum& s) {
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
float ri_angle_between(const float thetav, const float phiv, const float theta, const float phi) {
  const float cospsi = sinf(thetav) * sin(theta) * cosf(phi - phiv) + cosf(thetav) * cosf(theta);

  if (cospsi >= 1.f)
    return 0.f;
  if (cospsi <= -1.f)
    return M_PI;
  return acosf(cospsi);
}

__HD__
void sky_light_get_sky_spectral_radiance(const float theta, const float phi, Spectrum& spect, const SkyLight& sky_light) {
  const float theta_fin = min(theta, (float) ((M_PI * 0.5f) - 0.001f));
  const float gamma     = ri_angle_between(theta, phi, sky_light.theta_s, sky_light.phi_s);

  const float x = sky_light.zenith_x * sky_light_perez_base(sky_light.perez_x, theta_fin, gamma);
  const float y = sky_light.zenith_y * sky_light_perez_base(sky_light.perez_y, theta_fin, gamma);
  const float Y = sky_light.zenith_Y * sky_light_perez_base(sky_light.perez_Y, theta_fin, gamma);

  sky_light_chromaticity_to_spectrum(Y, x, y, spect);
}


__HD__
void sun_light_le(Spectrum& le, const Vector& dir, const SunLight& sun_light) {
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
void area_light_le(Spectrum& le, const Vector& wo, const Normal& light_normal, const AreaLightParam& mat) {
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
bool get_hit_point_information(const PtrFreeScene* const scene, Ray& ray, const RayHit& hit, Point& hit_point, Spectrum& surface_color, Normal& N, Normal& shade_N) {
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
void mesh_interpolate_color(const Spectrum* const colors, const Triangle* const triangles, const unsigned triangle_index, const float b1, const float b2, Spectrum& C) {
  const Triangle& triangle = triangles[triangle_index];
  const float b0 = 1.f - b1 - b2;

  C.r = b0 * colors[triangle.v[0]].r + b1 * colors[triangle.v[1]].r + b2 * colors[triangle.v[2]].r;
  C.g = b0 * colors[triangle.v[0]].g + b1 * colors[triangle.v[1]].g + b2 * colors[triangle.v[2]].g;
  C.b = b0 * colors[triangle.v[0]].b + b1 * colors[triangle.v[1]].b + b2 * colors[triangle.v[2]].b;
}

__HD__
void mesh_interpolate_normal(const Normal* const normals, const Triangle* const triangles, const unsigned triangle_index, const float b1, const float b2, Normal& N) {
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
void mesh_interpolate_UV(const UV* const uvs, const Triangle* const triangles, const unsigned triangle_index, const float b1, const float b2, UV& uv) {
  const Triangle& triangle = triangles[triangle_index];
  const float b0 = 1.f - b1 - b2;

  uv.u = b0 * uvs[triangle.v[0]].u + b1 * uvs[triangle.v[1]].u + b2 * uvs[triangle.v[2]].u;
  uv.v = b0 * uvs[triangle.v[0]].v + b1 * uvs[triangle.v[1]].v + b2 * uvs[triangle.v[2]].v;
}

__HD__
void generic_material_sample_f(const Material& mat, Vector& wo, Vector& wi, const Normal& N, const Normal& shade_N, const float u0, const float u1, const float u2, float& pdf, Spectrum& f, bool& specular_bounce) {
  switch (mat.type) {
    case MAT_AREALIGHT:
      break;
    case MAT_MATTE:       helpers::matte_material_sample_f              (mat.param.matte, wo, wi,    shade_N, u0, u1,     pdf, f, specular_bounce); break;
    case MAT_MIRROR:      helpers::mirror_material_sample_f            (mat.param.mirror, wo, wi,    shade_N,             pdf, f, specular_bounce); break;
    case MAT_GLASS:       helpers::glass_material_sample_f              (mat.param.glass, wo, wi, N, shade_N, u0,         pdf, f, specular_bounce); break;
    case MAT_MATTEMIRROR: helpers::matte_mirror_material_sample_f(mat.param.matte_mirror, wo, wi,    shade_N, u0, u1, u2, pdf, f, specular_bounce); break;
    case MAT_METAL:       helpers::metal_material_sample_f              (mat.param.metal, wo, wi,    shade_N, u0, u1,     pdf, f, specular_bounce); break;
    case MAT_MATTEMETAL:  helpers::matte_metal_material_sample_f  (mat.param.matte_metal, wo, wi,    shade_N, u0, u1, u2, pdf, f, specular_bounce); break;
    case MAT_ALLOY:       helpers::alloy_material_sample_f              (mat.param.alloy, wo, wi,    shade_N, u0, u1, u2, pdf, f, specular_bounce); break;
    case MAT_ARCHGLASS:   helpers::arch_glass_material_sample_f    (mat.param.arch_glass, wo, wi, N, shade_N, u0,         pdf, f, specular_bounce); break;
    case MAT_NULL:
      wi = - wo;
      specular_bounce = true;
      pdf = 1.f;
      break;
    default:
      specular_bounce = true;
      pdf = 0.f;
      break;
  }
}

__HD__
void matte_material_sample_f(const MatteParam& mat, const Vector& wo, Vector& wi, const Normal& shade_N, const float u0, const float u1, float& pdf, Spectrum& f, bool& specular_bounce) {
  Vector dir = CosineSampleHemisphere(u0, u1);
  pdf = dir.z * INV_PI;

  Vector v1, v2;
  CoordinateSystem((Vector) shade_N, &v1, &v2);

  wi.x = v1.x * dir.x + v2.x * dir.y + shade_N.x * dir.z;
  wi.y = v1.y * dir.x + v2.y * dir.y + shade_N.y * dir.z;
  wi.z = v1.z * dir.x + v2.z * dir.y + shade_N.z * dir.z;

  const float dp = Dot(shade_N, wi);

  if (dp <= 0.0001f) {
    pdf = 0.f;
  } else {
    f.r = mat.kd.r * INV_PI;
    f.g = mat.kd.g * INV_PI;
    f.b = mat.kd.b * INV_PI;
    pdf /= dp;
  }
  specular_bounce = false;
}

__HD__
void mirror_material_sample_f(const MirrorParam& mat, const Vector& wo, Vector& wi, const Normal& shade_N, float& pdf, Spectrum& f, bool& specular_bounce) {
  const float k = 2.f * Dot(shade_N, wo);
  wi.x = k * shade_N.x - wo.x;
  wi.y = k * shade_N.y - wo.y;
  wi.z = k * shade_N.z - wo.z;

  pdf = 1.f;
  f.r = mat.kr.r;
  f.g = mat.kr.g;
  f.b = mat.kr.b;
  specular_bounce = mat.specular_bounce;
}

__HD__
void glass_material_sample_f(const GlassParam& mat, const Vector& wo, Vector& wi, const Normal& N, const Normal& shade_N, const float u0, float& pdf, Spectrum& f, bool& specular_bounce) {
  const float k = 2.f * Dot(N, wo);
  Vector refl_dir;
  refl_dir.x = k * N.x - wo.x;
  refl_dir.y = k * N.y - wo.y;
  refl_dir.z = k * N.z - wo.z;

  const bool into = (Dot(N, shade_N) > 0.f);
  const float nc = mat.outside_ior;
  const float nt = mat.ior;
  const float nnt = into ? (nc / nt) : (nt / nc);
  const float ddn = -Dot(wo, shade_N);
  const float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

  if (cos2t < 0.f) {
    wi = refl_dir;
    pdf = 1.f;
    f.r = mat.refl.r;
    f.g = mat.refl.g;
    f.b = mat.refl.b;
    specular_bounce = mat.reflection_specular_bounce;
  } else {
    const float kk = (into ? 1.f : -1.f) * (ddn * nnt + sqrt(cos2t));
    Vector nkk = (Vector) N;
    nkk *= kk;

    Vector trans_dir = -nnt * wo - nkk;
    Normalize(trans_dir);

    const float c = 1.f - (into ? -ddn : Dot(trans_dir, N));
    const float R0 = mat.R0;
    const float Re = R0 + (1.f - R0) * c * c * c * c * c;
    const float Tr = 1.f - Re;
    const float P = .25f + .5f * Re;

    if (Tr == 0.f) {
      if (Re == 0.f)
        pdf = 0.f;
      else {
        wi = refl_dir;
        pdf = 1.f;
        f.r = mat.refl.r;
        f.g = mat.refl.g;
        f.b = mat.refl.b;
        specular_bounce = mat.reflection_specular_bounce;
      }

    } else if (Re == 0.f) {
      wi = trans_dir;
      pdf = 1.f;
      f.r = mat.refrct.r;
      f.g = mat.refrct.g;
      f.b = mat.refrct.b;
      specular_bounce = mat.transmission_specular_bounce;

    } else if (u0 < P) {
      wi = refl_dir;
      pdf = P / Re;
      f.r = mat.refl.r;
      f.g = mat.refl.g;
      f.b = mat.refl.b;
      specular_bounce = mat.reflection_specular_bounce;

    } else {
      wi = trans_dir;
      pdf = (1.f - P) / Tr;
      f.r = mat.refrct.r;
      f.g = mat.refrct.g;
      f.b = mat.refrct.b;
      specular_bounce = mat.transmission_specular_bounce;
    }
  }
}

__HD__
void matte_mirror_material_sample_f(const MatteMirrorParam& mat, const Vector& wo, Vector& wi, const Normal& shade_N, const float u0, const float u1, const float u2, float& pdf, Spectrum& f, bool& specular_bounce) {
  const float tot_filter = mat.tot_filter;
  const float comp = u2 * tot_filter;

  float mpdf;
  if (comp > mat.matte_filter) {
    mirror_material_sample_f(mat.mirror, wo, wi, shade_N, pdf, f, specular_bounce);
    mpdf = mat.mirror_pdf;
  } else {
    matte_material_sample_f(mat.matte, wo, wi, shade_N, u0, u1, pdf, f, specular_bounce);
    mpdf = mat.matte_pdf;
  }

  pdf *= mpdf;
}

__HD__
void metal_material_sample_f(const MetalParam& mat, const Vector& wo, Vector& wi, const Normal& shade_N, const float u0, const float u1, float& pdf, Spectrum& f, bool& specular_bounce) {
  glossy_reflection(wo, wi, mat.exp, shade_N, u0, u1);

  if (Dot(wi, shade_N) > 0.f) {
    pdf = 1.f;
    f.r = mat.kr.r;
    f.g = mat.kr.g;
    f.b = mat.kr.b;
    specular_bounce = mat.specular_bounce;
  } else {
    pdf = 0.f;
  }
}

__HD__
void matte_metal_material_sample_f(const MatteMetalParam& mat, const Vector& wo, Vector& wi, const Normal& shade_N, const float u0, const float u1, const float u2, float& pdf, Spectrum& f, bool& specular_bounce) {
  const float tot_filter = mat.tot_filter;
  const float comp = u2 * tot_filter;

  float mpdf;
  if (comp > mat.matte_filter) {
    metal_material_sample_f(mat.metal, wo, wi, shade_N, u0, u1, pdf, f, specular_bounce);
    mpdf = mat.metal_pdf;
  } else {
    matte_material_sample_f(mat.matte, wo, wi, shade_N, u0, u1, pdf, f, specular_bounce);
    mpdf = mat.matte_pdf;
  }
  pdf *= mpdf;
}

__HD__
void alloy_material_sample_f(const AlloyParam& mat, const Vector& wo, Vector& wi, const Normal& shade_N, const float u0, const float u1, const float u2, float& pdf, Spectrum& f, bool& specular_bounce) {
  const float c = 1.f - Dot(wo, shade_N);
  const float R0 = mat.R0;
  const float Re = R0 + (1.f - R0) * c * c * c * c * c;
  const float P = .25f + .5f * Re;

  if (u2 <= P) {
    glossy_reflection(wo, wi, mat.exp, shade_N, u0, u1);
    pdf = P / Re;
    f.r = mat.refl.r * Re;
    f.g = mat.refl.g * Re;
    f.b = mat.refl.b * Re;
    specular_bounce = mat.specular_bounce;
  } else {
    Vector dir = CosineSampleHemisphere(u0, u1);
    pdf = dir.z * INV_PI;

    Vector v1, v2;
    CoordinateSystem((Vector) shade_N, &v1, &v2);

    wi.x = v1.x * dir.x + v2.x * dir.y + shade_N.x * dir.z;
    wi.y = v1.y * dir.x + v2.y * dir.y + shade_N.y * dir.z;
    wi.z = v1.z * dir.x + v2.z * dir.y + shade_N.z * dir.z;

    if (dir.z <= 0.0001f)
      pdf = 0.f;
    else {
      const float iRe = 1.f - Re;
      const float k =  ( 1.f - P) / iRe;
      pdf *= k;
      f.r = mat.diff.r * iRe;
      f.g = mat.diff.g * iRe;
      f.b = mat.diff.b * iRe;
      specular_bounce = false;
    }
  }
}

__HD__
void arch_glass_material_sample_f(const ArchGlassParam& mat, const Vector& wo, Vector& wi, const Normal& N, const Normal& shade_N, const float u0, float& pdf, Spectrum& f, bool& specular_bounce) {
  const bool into = (Dot(N, shade_N) > 0.f);

  if (!into) {
    wi = -wo;
    pdf = 1.f;
    f.r = mat.refrct.r;
    f.g = mat.refrct.g;
    f.b = mat.refrct.b;
    specular_bounce = mat.transmission_specular_bounce;
  } else {
    const float comp = u0 * mat.tot_filter;

    if (comp > mat.trans_filter) {
      const float k = 2.f * Dot(N, wo);
      wi.x = k * N.x - wo.x;
      wi.y = k * N.y - wo.y;
      wi.z = k * N.z - wo.z;

      pdf = mat.refl_pdf;
      f.r = mat.refl.r;
      f.g = mat.refl.g;
      f.b = mat.refl.b;
      specular_bounce = mat.reflection_specular_bounce;
    } else {
      wi = -wo;
      f.r = mat.refrct.r;
      f.g = mat.refrct.g;
      f.b = mat.refrct.b;
      specular_bounce = mat.transmission_specular_bounce;
    }
  }
}

__HD__
void glossy_reflection(const Vector& wo, Vector& wi, const float exponent, const Normal& shade_N, const float u0, const float u1) {
  const float phi = 2.f * M_PI * u0;
  const float cos_theta = powf(1.f - u1, exponent);
  const float sin_theta = sqrtf(Max(0.f, 1.f - cos_theta * cos_theta));
  const float x = cosf(phi) * sin_theta;
  const float y = sinf(phi) * sin_theta;
  const float z = cos_theta;

  const Vector dir = -wo;
  const float dp = Dot(shade_N, dir);
  const Vector w = dir - (2.f * dp) * Vector(shade_N);

  Vector u;
  if (fabsf(shade_N.x) > .1f) {
    const Vector a(0.f, 1.f, 0.f);
    u = Cross(a, w);
  } else {
    const Vector a(1.f, 0.f, 0.f);
    u = Cross(a, w);
  }
  u = Normalize(u);
  Vector v = Cross(w, u);

  wi = x * u + y * v + z * w;
}

__HD__
LightType sample_all_lights(const float u, const float lights_count, const InfiniteLight& infinite_light, const SunLight& sun_light, const SkyLight& sky_light, float& pdf, int& light_index, const bool skip_inifinite_ligth = false) {

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
void infinite_light_sample_l(const float u0, const float u1, const float u2, const float u3, const InfiniteLigh& infinite_light, const Spectrum* const infinite_light_map, const BSphere& bsphere, float& pdf, Ray& ray, Spectrum& f) {
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
void sun_light_sample_l(const float u0, const float u1, const SunLight& sun_light, const Point& hit_point, float& pdf, Ray& shadow_ray, Spectrum& f) {
  Vector wi = UniformSampleCone(u0, u1, sun_light.cos_theta_max, sun_light.x, sun_light.y, syn_light.dir);

  shadow_ray.o = hit_point;
  shadow_ray.d = wi;
  shadow_ray.min_t = RAY_EPSILON;
  shadow_ray.max_t = FLT_MAX;

  f = sun_light.color;
  pdf = UniformConePdf(sun_light.cos_theta_max);
}

__HD__
void sun_light_sample_l(const float u0, const float u1, const float u2, const float u3, const SunLigh& sun_light, const BSphere& bsphere, float& pdf, Ray& ray, Spectrum& f) {
  const float rad = bsphere.rad * 1.01f;

  float d1, d2;
  ConcentricSampleDisk(u0, u1, &d1, &d2);
  const Point p_disk = bsphere.center + rad * (d1 * sun_light.x + d2 * sun_light.y);

  ray = Ray(p_disk + rad * sun_light.dir, -UniformSampleCone(u2, u3, sun_light.cos_theta_max, sun_light.x, sun_light.y, sun_light.dir));
  pdf = UniformCodePdf(sun_light.cos_theta_max) / (M_PI * rad * rad);
}

}

} }
