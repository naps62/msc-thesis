/*
 * material.h
 *
 *  Created on: Mar 20, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_TYPES_MATERIAL_H_
#define _PPM_TYPES_MATERIAL_H_

namespace ppm {

struct MatteParam {
	Spectrum kd;
};

struct MirrorParam {
	Spectrum kr;
	int specular_bounce;
};

struct GlassParam {
	Spectrum refl, refrct;
	float outside_ior, ior;
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
	float exp;
	int specular_bounce;
};

struct MatteMetalParam {
	MatteParam matte;
	MetalParam metal;
	float matte_filter, tot_filter, matte_pdf, metal_pdf;
};

struct AlloyParam {
	Spectrum diff, refl;
	float exp;
	float R0;
	int specular_bounce;
};

struct ArchGlassParam {
	Spectrum refl, refrct;
	float trans_filter, tot_filter, refl_pdf, trans_pdf;
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
