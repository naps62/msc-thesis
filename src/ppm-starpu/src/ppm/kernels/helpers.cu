#include "ppm/kernels/helpers.cuh"

namespace ppm { namespace kernels {

namespace helpers {

__HD__
void tex_map_get_texel(const Spectrum* const pixels, const unsigned width, const unsigned height, const int s, const int t, Spectrum* const color) {
  const unsigned u = Mod(s, width);
  const unsigned v = Mod(s, height);

  const Spectrum& pixel = pixels[v * width + u];

  color->r = pixel.r;
  color->g = pixel.g;
  color->b = pixel.b;
}

__HD__
void tex_map_get_color(const Spectrum* const pixels, const unsigned width, const unsigned height, const float u, const float v, Spectrum* const color) {
  const float s = u * width  - 0.5f;
  const float t = v * height - 0.5f;

  const int s0 = (int) floor(s);
  const int t0 = (int) floor(t);

  const float ds = s - s0;
  const float dt = t - t0;

  const float ids = 1.f - ds;
  const float idt = 1.f - dt;

  Spectrum c0, c1, c2, c3;
  tex_map_get_texel(pixels, width, height, s0,     t0,     &c0);
  tex_map_get_texel(pixels, width, height, s0,     t0 + 1, &c1);
  tex_map_get_texel(pixels, width, height, s0 + 1, t0,     &c2);
  tex_map_get_texel(pixels, width, height, s0 + 1, t0 + 1, &c3);

  const float k0 = ids * idt;
  const float k1 = ids * dt;
  const float k2 = ds  * idt;
  const float k3 = ds  * dt;

  color->r = k0 * c0.r + k1 * c1.r + k2 * c2.r + k3 * c3.r;
  color->g = k0 * c0.g + k1 * c1.g + k2 * c2.g + k3 * c3.g;
  color->b = k0 * c0.b + k1 * c1.b + k2 * c2.b + k3 * c3.b;
}


__HD__
void infinite_light_le (Spectrum* const le, const Vector* const dir, const InfiniteLight* const infinite_light, const Spectrum* const infinite_light_map) {

  const float u = 1.f - SphericalPhi(*dir)   * INV_TWOPI + infinite_light->shiftU;
  const float v =       SphericalTheta(*dir) * INV_PI    + infinite_light->shiftV;

  tex_map_get_color(infinite_light_map, infinite_light->width, infinite_light->height, u, v, le);

  le->r *= infinite_light->gain.r;
  le->g *= infinite_light->gain.g;
  le->b *= infinite_light->gain.b;
}


__HD__
void sky_light_le(Spectrum* const f, const Vector* const dir, const SkyLight* const sky_light) {

  const float theta = SphericalTheta(*dir);
  const float phi   = SphericalPhi(*dir);

  Spectrum s;
  sky_light_get_sky_spectral_radiance(theta, phi, &s, sky_light);

  f->r = sky_light->gain.r * s.r;
  f->g = sky_light->gain.g * s.g;
  f->b = sky_light->gain.b * s.b;
}

__HD__
float sky_light_perez_base(const float* const lam, const float theta, const float gamma) {
  return (1.f + lam[1] * exp(lam[2] / cos(theta))) * (1.f + lam[3] * exp(lam[4] * gamma) + lam[5] * cos(gamma) * cos(gamma));
}

__HD__
void sky_light_chromaticity_to_spectrum(const float Y, const float x, const float y, Spectrum* const s) {
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
  s->r =  3.2410f * X - 1.5374f * Y - 0.4986f * Z;
  s->g = -0.9692f * X + 1.8760f * Y + 0.0416f * Z;
  s->b =  0.0556f * X - 0.2040f * Y + 1.0570f * Z;
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
void sky_light_get_sky_spectral_radiance(const float theta, const float phi, Spectrum* const spect, const SkyLight* const sky_light) {
  const float theta_fin = min(theta, (float) ((M_PI * 0.5f) - 0.001f));
  const float gamma     = ri_angle_between(theta, phi, sky_light->theta_s, sky_light->phi_s);

  const float x = sky_light->zenith_x * sky_light_perez_base(sky_light->perez_x, theta_fin, gamma);
  const float y = sky_light->zenith_y * sky_light_perez_base(sky_light->perez_y, theta_fin, gamma);
  const float Y = sky_light->zenith_Y * sky_light_perez_base(sky_light->perez_Y, theta_fin, gamma);

  sky_light_chromaticity_to_spectrum(Y, x, y, spect);
}


__HD__
void sun_light_le(Spectrum* const le, const Vector* const dir, const SunLight* const sun_light) {
  const float cos_theta_max = sun_light->cos_theta_max;
  const Vector sun_dir = sun_light->dir;

  if ((cos_theta_max < 1.f) && (Dot(*dir, sun_dir) > cos_theta_max))
    *le = sun_light->color;
  else {
    le->r = 0.f;
    le->g = 0.f;
    le->b = 0.f;
  }
}

}

} }
