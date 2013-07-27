#include "ppm/kernels/helpers.cuh"
#include "ppm/ptrfree_hash_grid.h"
#include "ppm/math.h"
#include <limits>
#include <cfloat>

namespace ppm { namespace kernels {

namespace helpers {


__HD__
void generic_material_sample_f(
    const Material& mat,
    Vector& wo,
    Vector& wi,
    const Normal& N,
    const Normal& shade_N,
    const float u0,
    const float u1,
    const float u2,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {

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
void matte_material_sample_f(
    const MatteParam& mat,
    const Vector& wo,
    Vector& wi,
    const Normal& shade_N,
    const float u0,
    const float u1,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {

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
void mirror_material_sample_f(
    const MirrorParam& mat,
    const Vector& wo,
    Vector& wi,
    const Normal& shade_N,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {

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
void glass_material_sample_f(
    const GlassParam& mat,
    const Vector& wo,
    Vector& wi,
    const Normal& N,
    const Normal& shade_N,
    const float u0,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {

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
void matte_mirror_material_sample_f(
    const MatteMirrorParam& mat,
    const Vector& wo,
    Vector& wi,
    const Normal& shade_N,
    const float u0,
    const float u1,
    const float u2,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {

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
void metal_material_sample_f(
    const MetalParam& mat,
    const Vector& wo,
    Vector& wi,
    const Normal& shade_N,
    const float u0,
    const float u1,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {
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
void matte_metal_material_sample_f(
    const MatteMetalParam& mat,
    const Vector& wo,
    Vector& wi,
    const Normal& shade_N,
    const float u0,
    const float u1,
    const float u2,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {
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
void alloy_material_sample_f(
    const AlloyParam& mat,
    const Vector& wo,
    Vector& wi,
    const Normal& shade_N,
    const float u0,
    const float u1,
    const float u2,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {

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
void arch_glass_material_sample_f(
    const ArchGlassParam& mat,
    const Vector& wo,
    Vector& wi,
    const Normal& N,
    const Normal& shade_N,
    const float u0,
    float& pdf,
    Spectrum& f,
    bool& specular_bounce) {

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
      pdf = mat.trans_pdf;
      f.r = mat.refrct.r;
      f.g = mat.refrct.g;
      f.b = mat.refrct.b;
      specular_bounce = mat.transmission_specular_bounce;
    }
  }
}

__HD__
void glossy_reflection(
    const Vector& wo,
    Vector& wi,
    const float exponent,
    const Normal& shade_N,
    const float u0,
    const float u1) {

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


__HD__ void matte_f(
    const MatteParam& mat,
    Spectrum& f) {

  f.r = mat.kd.r * INV_PI;
  f.g = mat.kd.g * INV_PI;
  f.b = mat.kd.b * INV_PI;
}

__HD__ void matte_mirror_f(
    const MatteMirrorParam& mat,
    Spectrum& f) {
  matte_f(mat.matte, f);
  f *= mat.matte_pdf;
}

__HD__ void matte_metal_f(
    const MatteMetalParam& mat,
    Spectrum& f) {
  matte_f(mat.matte, f);
  f *= mat.matte_pdf;
}

__HD__ void alloy_f(
    const AlloyParam& mat,
    const Vector& wo,
    const Normal& N,
    Spectrum& f) {

  const float c  = 1.f - Dot(wo, N);
  const float Re = mat.R0 + (1.f - mat.R0) * c * c * c * c * c;
  const float P  = .25f + .5f * Re;

  f.r = mat.diff.r * INV_PI;
  f.g = mat.diff.g * INV_PI;
  f.b = mat.diff.b * INV_PI;

  f *= (1.f - Re) / (1.f - P);
}

template<class T> __host__ __device__ void my_atomic_add(T* var, T inc) {
#ifdef __CUDA_ARCH__
  atomicAdd(var, inc);
#else
  __sync_fetch_and_add(var, inc);
#endif
}


__HD__ void add_flux(
    const PtrFreeHashGrid* const hash_grid,
    const PtrFreeScene* const scene,
    const Point& hit_point,
    const Normal& shade_N,
    const Vector& wi,
    const Spectrum& photon_flux,
    HitPointStaticInfo* const hit_points_info,
    HitPoint* const hit_points) {

  const Vector hh = (hit_point - hash_grid->bbox.pMin) * hash_grid->inv_cell_size;
  const int ix = abs(int(hh.x));
  const int iy = abs(int(hh.y));
  const int iz = abs(int(hh.z));

  unsigned grid_index = hash(ix, iy, iz, hash_grid->size);
  unsigned length = hash_grid->lengths[grid_index];

  if (length > 0) {
    unsigned local_list = hash_grid->lists_index[grid_index];
    for(unsigned i = local_list; i < local_list + length; ++i) {
      unsigned hit_point_index = hash_grid->lists[i];
      HitPointStaticInfo& ihp = hit_points_info[hit_point_index];
      HitPoint& hp = hit_points[hit_point_index];

      Vector v = ihp.position - hit_point;

      //if ((Dot(ihp.normal, shade_N) <= 0.5f) || (Dot(v, v) > hp.accum_photon_radius2))
      if (DistanceSquared(ihp.position, hit_point) > hp.accum_photon_radius2 || Dot(ihp.normal, wi) <= 0.0001f)
        continue;

      my_atomic_add(&hp.accum_photon_count, (unsigned)1);

      Spectrum f;

      Material& hit_point_mat = scene->materials[ihp.material_ss];
      switch(hit_point_mat.type) {
        case MAT_MATTE: matte_f(hit_point_mat.param.matte, f);
                        break;
        case MAT_MATTEMIRROR: matte_mirror_f(hit_point_mat.param.matte_mirror, f);
                              break;
        case MAT_MATTEMETAL:  matte_metal_f(hit_point_mat.param.matte_metal, f);
                              break;
        case MAT_ALLOY: alloy_f(hit_point_mat.param.alloy, ihp.wo, shade_N, f);
                        break;
      }

      Spectrum flux = photon_flux * AbsDot(shade_N, wi) * ihp.throughput * f;

#ifdef __CUDA_ARCH__
      my_atomic_add(&hp.accum_reflected_flux.r, flux.r);
      my_atomic_add(&hp.accum_reflected_flux.g, flux.g);
      my_atomic_add(&hp.accum_reflected_flux.b, flux.b);
#else
#pragma omp critical
      {
        hp.accum_reflected_flux = hp.accum_reflected_flux + flux;
      }
#endif
    }
  }
}

__HD__ unsigned hash(const int ix, const int iy, const int iz, unsigned size) {
  return (unsigned) ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) % size;
}

}

} }
