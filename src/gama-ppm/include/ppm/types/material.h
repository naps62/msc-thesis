/*
 * material.h
 *
 *  Created on: Mar 13, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_TYPES_MATERIAL_H_
#define _PPM_TYPES_MATERIAL_H_

#include <cstring>

namespace ppm {

struct MatteParam {
	Spectrum kd;
};

struct AreaLightParam {
	Spectrum gain;
//	float gain_r, gain_g, gain_b;
};

struct MirrorParam {
	Spectrum kr;
//	float r, g, b;
	int specular_bounce;
};

struct GlassParam {
	Spectrum refl, refrct;
//	float refl_r, refl_g, refl_b;
//	float refrct_r, refrct_g, refrct_b;
	float ouside_ior, ior; // TODO is this "outside_ior"?
	float R0;
	int reflection_specular_bounce, transmission_specular_bounce;
};

struct MatteMirrorParam {
	MatteParam matte;
	MirrorParam mirror;
	float matte_filter, tot_filter, matte_pdf, mirror_pdf;
};

struct MetalParam {
	Spectrum kr;
//	float r, g, b;
	float exponent;
	int specular_bounce;
};

struct MatteMetalParam {
	MatteParam matte;
	MetalParam metal;
	float matte_filter, tot_filter, matte_pdf, metal_pdf;
};

struct AlloyParam {
	Spectrum diff, refl;
//	float diff_r, diff_g, diff_b;
//	float refl_r, refl_g, refl_b;
	float exponent;
	float R0;
	int specular_bounce;
};

struct ArchGlassParam {
	Spectrum refl, refrct;
//	float refl_r, refl_g, refl_b;
//	float refrct_r, refrct_g, refrct_b;
	float trans_filter, tot_filter, refl_filter, trans_pdf;
	bool reflection_specular_bounce, transmission_specular_bounce;
};

struct Material {
	uint type;
	bool diffuse;
	bool specular;
	union {
		MatteParam matte;
		AreaLightParam area_light;
		MirrorParam mirror;
		GlassParam glass;
		MatteMirrorParam matte_mirror;
		MetalParam metal;
		MatteMetalParam matte_metal;
		AlloyParam alloy;
		ArchGlassParam arch_glass;
	} param;

};

}

#endif // _PPM_TYPES_MATERIAL_H_
