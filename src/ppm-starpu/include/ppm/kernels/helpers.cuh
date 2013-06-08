#ifndef _PPM_KERNELS_HELPERS_H_
#define _PPM_KERNELS_HELPERS_H_

#include "ppm/types.h"

namespace ppm { namespace kernels {

namespace helpers {

__HD__ void tex_map_get_texel(const Spectrum* const pixels, const unsigned width, const unsigned height, const int s, const int t, Spectrum* const color);
__HD__ void tex_map_get_color(const Spectrum* const pixels, const unsigned width, const unsigned height, const float u, const float v, Spectrum* const color);

__HD__ void infinite_light_le (Spectrum* const le, const Vector* const dir, const InfiniteLight* const infinite_light, const Spectrum* const infinite_light_map);
__HD__ void sky_light_le      (Spectrum* const le, const Vector* const dir, const SkyLight* const sky_light);
__HD__ float sky_light_perez_base(const float* const lam, const float theta, const float gamma);
__HD__ void sky_light_chromaticity_to_spectrum(const float Y, const float x, const float y, Spectrum* const s);
__HD__ float ri_angle_between(const float thetav, const float phiv, const float theta, const float phi);
__HD__ void sky_light_get_sky_spectral_radiance(const float theta, const float phi, Spectrum* const spect, const SkyLight* const sky_light);
__HD__ void sky_light_get_sky_spectral_radiance(const float theta, const float phi, const Spectrum* const spect, const SkyLight* const sky_light);
__HD__ void sun_light_le(Spectrum* const le, const Vector* const dir, const SunLight* const sun_light);

}

} }

#endif // _PPM_KERNELS_HELPERS_H_
