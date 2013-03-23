/*
 * material.h
 *
 *  Created on: Mar 20, 2013
 *      Author: Miguel Palhas
 */

#ifndef _PPM_TYPES_MATERIAL_H_
#define _PPM_TYPES_MATERIAL_H_

#include "ppm/types.h"

namespace ppm {

typedef enum {
	MAT_MATTE,
	MAT_AREALIGHT,
	MAT_MIRROR,
	MAT_GLASS,
	MAT_MATTEMIRROR,
	MAT_METAL,
	MAT_MATTEMETAL,
	MAT_ALLOY,
	MAT_ARCHGLASS,

	MAT_MAX
} CompiledMaterials_e;

struct MatteParam {
	Color kd;
};

ostream& operator<< (ostream& os, const MatteParam& m) {
	return os << "MatteParam[" << m.kd << "]";
}

struct AreaLightParam {
	Color gain;
};

ostream& operator<< (ostream& os, const AreaLightParam& m) {
	return os << "AreaLightParam[" << m.gain << "]";
}

struct MirrorParam {
	Color kr;
	int specular_bounce;
};

ostream& operator<< (ostream& os, const MirrorParam& m) {
	return os << "MirrorParam[" << m.kr << ", " << m.specular_bounce << "]";
}

struct GlassParam {
	Color refl, refrct;
	float outside_ior, ior;
	float R0;
	int reflection_specular_bounce, transmission_specular_bounce;
};

ostream& operator<< (ostream& os, const GlassParam& m) {
	return os << "GlassParam[" << m.refl << ", " << m.refrct << ", " << m.outside_ior << "; " << m.ior
			  << ", " << m.R0 << ", " << m.reflection_specular_bounce << ", " << m.transmission_specular_bounce << "]";
}

struct MatteMirrorParam {
	MatteParam matte;
	MirrorParam mirror;
	float matte_filter, tot_filter, matte_pdf, mirror_pdf;
};

ostream& operator<< (ostream& os, const MatteMirrorParam& m) {
	return os << "MatteMirrorParam[" << m.matte << ", " << m.mirror << ", "
			  << m.matte_filter << ", " << m.tot_filter << ", " << m.matte_pdf << ", " << m.mirror_pdf << "]";
}

struct MetalParam {
	Color kr;
	float exp;
	int specular_bounce;
};

ostream& operator<< (ostream& os, const MetalParam& m) {
	return os << "MetalParam[" << m.kr << ", " << m.exp << ", " << m.specular_bounce << "]";
}

struct MatteMetalParam {
	MatteParam matte;
	MetalParam metal;
	float matte_filter, tot_filter, matte_pdf, metal_pdf;
};

ostream& operator<< (ostream& os, const MatteMetalParam& m) {
	return os << "MatteMetalParam[" << m.matte << ", " << m.metal << ", "
			  << m.matte_filter << ", " << m.tot_filter << ", " << m.matte_pdf << ", " << m.metal_pdf << "]";
}

struct AlloyParam {
	Color diff, refl;
	float exp;
	float R0;
	int specular_bounce;
};

ostream& operator<< (ostream& os, const AlloyParam& m) {
	return os << "AlloyParam[" << m.diff << ", " << m.refl << ", " << m.exp << ", " << m.R0 << ", " << m.specular_bounce << "]";
}

struct ArchGlassParam {
	Color refl, refrct;
	float trans_filter, tot_filter, refl_pdf, trans_pdf;
	bool reflection_specular_bounce, transmission_specular_bounce;
};

ostream& operator<< (ostream& os, const ArchGlassParam& m) {
	return os << "ArchGlassParam[" << m.refl << ", " << m.refrct << ", "
			  << m.trans_filter << ", " << m.tot_filter << ", " << m.refl_pdf << ", " << m.trans_pdf << ", "
			  << m.reflection_specular_bounce << ", " << m.transmission_specular_bounce << "]";
}

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

ostream& operator<< (ostream& os, const Material& m) {
	switch (m.type) {
	case MAT_MATTE:       return os << static_cast<MatteParam>(m); break;
	case MAT_AREALIGHT:   return os << static_cast<AreaLightParam>(m); break;
	case MAT_MIRROR:      return os << static_cast<MirrorParam>(m); break;
	case MAT_GLASS:       return os << static_cast<GlassParam>(m); break;
	case MAT_MATTEMIRROR: return os << static_cast<MatteMirrorParam>(m); break;
	case MAT_METAL:       return os << static_cast<MetalParam>(m); break;
	case MAT_MATTEMETAL:  return os << static_cast<MatteMetalParam>(m); break;
	case MAT_ALLOY:       return os << static_cast<AlloyParam>(m); break;
	case MAT_ARCHGLASS:   return os << static_cast<ArchGlassParam>(m); break;
	}
}

}

#endif // _PPM_TYPES_MATERIAL_H_
