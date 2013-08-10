#include "pointerfreescene_types.h"
#include <ostream>
#include "dumps.h"

namespace POINTERFREESCENE {

// Mesh
ostream& operator<< (ostream& os, const Mesh& m) {
	os  << "Mesh["
		<< "verts_offset("  << m.vertsOffset << "), "
		<< "tris_offset("   << m.trisOffset << "), "
		<< "colors_offset(" << m.colorsOffset << "), "
		<< "has_colors("    << m.hasColors << "), "
		<< "trans:";
		os << "Matrix4x4[ ";
		for (int i = 0; i < 4; ++i) {
			os << "[ ";
			for (int j = 0; j < 4; ++j) {
				os << m.trans[i][j];
				if (j != 3) os << ", ";
			}
			os << " ]";
		}
		os << "], inv_trans:Matrix4x4[ ";
		for (int i = 0; i < 4; ++i) {
			os << "[ ";
			for (int j = 0; j < 4; ++j) {
				os << m.invTrans[i][j];
				if (j != 3) os << ", ";
			}
			os << " ]";
		}
		return os << "]]";
}

// Material
ostream& operator<< (ostream& os, const Material& m) {
	os << "Material[" << m.type << ", " << m.difuse << ", " << m.specular << ", ";
	switch (m.type) {
	case MAT_MATTE:       os << m.param.matte;        break;
	case MAT_AREALIGHT:   os << m.param.areaLight;   break;
	case MAT_MIRROR:      os << m.param.mirror;       break;
	case MAT_GLASS:       os << m.param.glass;        break;
	case MAT_MATTEMIRROR: os << m.param.matteMirror; break;
	case MAT_METAL:       os << m.param.metal;        break;
	case MAT_MATTEMETAL:  os << m.param.matteMetal;  break;
	case MAT_ALLOY:       os << m.param.alloy;        break;
	case MAT_ARCHGLASS:   os << m.param.archGlass;   break;
	}
	return os << "]";
}

// MatteParam
ostream& operator<< (ostream& os, const MatteParam& m) {
	return os << "MatteParam[Color[" << m.r << ", " << m.g << ", " << m.b << "]]";
}
// AreaLightParam
ostream& operator<< (ostream& os, const AreaLightParam& m) {
	return os << "AreaLightParam[Color[" << m.gain_r << ", " << m.gain_g << ", " << m.gain_b << "]]";
}
// MirrorParam
ostream& operator<< (ostream& os, const MirrorParam& m) {
	return os << "MirrorParam[Color[" << m.r << ", " << m.g << ", " << m.b << "], " << m.specularBounce << "]";
}
// GlassParam
ostream& operator<< (ostream& os, const GlassParam& m) {
	return os << "GlassParam[Color[" << m.refl_r << ", " << m.refl_g << ", " << m.refl_b << ", " << "], "
			<< "Color[" << m.refrct_r << ", " << m.refrct_g << ", " << m.refrct_b << "], "
			<< m.ousideIor << "; " << m.ior
			  << ", " << m.R0 << ", " << m.reflectionSpecularBounce << ", " << m.transmitionSpecularBounce << "]";
}
// MatteMirrorParam
ostream& operator<< (ostream& os, const MatteMirrorParam& m) {
	return os << "MatteMirrorParam[" << m.matte << ", " << m.mirror << ", "
			  << m.matteFilter << ", " << m.totFilter << ", " << m.mattePdf << ", " << m.mirrorPdf << "]";
}
// MetalParam
ostream& operator<< (ostream& os, const MetalParam& m) {
	return os << "MetalParam[Color[" << m.r << ", " << m.g << ", " << m.b << "], " << m.exponent << ", " << m.specularBounce << "]";
}
// MatteMetalParam
ostream& operator<< (ostream& os, const MatteMetalParam& m) {
	return os << "MatteMetalParam[" << m.matte << ", " << m.metal << ", "
			  << m.matteFilter << ", " << m.totFilter << ", " << m.mattePdf << ", " << m.metalPdf << "]";
}
// AlloyParam
ostream& operator<< (ostream& os, const AlloyParam& m) {
	return os << "AlloyParam[Color[" << m.diff_r << ", " << m.diff_g << ", " << m.diff_b << ", " << "], Color["
			<< m.refl_r << ", " << m.refl_g << ", " << m.refl_b << ", " << "], " << m.exponent << ", " << m.R0 << ", " << m.specularBounce << "]";
}

// ArchGlassParam
ostream& operator<< (ostream& os, const ArchGlassParam& m) {
	return os << "ArchGlassParam[Color[" << m.refl_r << ", " << m.refl_g << ", " << m.refl_b << ", " << "], "
			<< "Color[" << m.refrct_r << ", " << m.refrct_g << ", " << m.refrct_b << "], "
			  << m.transFilter << ", " << m.totFilter << ", " << m.reflPdf << ", " << m.transPdf << ", "
			  << m.reflectionSpecularBounce << ", " << m.transmitionSpecularBounce << "]";
}


// TriangleLight
ostream& operator<< (ostream& os, const TriangleLight& l) {
	return os << "TriangleLight[" << l.v0 << ", " << l.v1 << ", " << l.v2 << ", "
			  << l.normal << ", " << l.area << ", Spectrum[" << l.gain_r << ", " << l.gain_g << ", " << l.gain_b << "], " << l.meshIndex << ", " << l.triIndex << "]";
}

// InfiniteLight
ostream& operator<< (ostream& os, const InfiniteLight& l) {
	return os << "InfiniteLight[" << 1 << ", " << l.shiftU << "; " << l.shiftV << ", " << l.gain
			  << ", " << l.width << ", " << l.height << "]";
}

// SunLight
ostream& operator<< (ostream& os, const SunLight& l) {
	return os << "SunLight[" << 1 << ", " << l.sundir << ", " << l.gain
			  << ", " << l.turbidity << ", " << l.relSize << ", " << l.x << ", " << l.y
			  << ", " << l.cosThetaMax << ", " << l.suncolor << "]";
}

// SkyLight
ostream& operator<< (ostream& os, const SkyLight& l) {
	return os << "SkyLight[" << 1 << ", " << l.gain << ", " << l.thetaS << ", "
			  << l.phiS << "; " << l.zenith_Y << ", " << l.zenith_x << "; " << l.zenith_Y << ", "
			  << "perez_Y["
			      << l.perez_Y[0] << ' ' << l.perez_Y[1] << ' ' << l.perez_Y[2] << ' ' << l.perez_Y[3] << ' ' << l.perez_Y[4] << ' ' << l.perez_Y[5] << "], "
			  << "perez_Y["
			      << l.perez_x[0] << ' ' << l.perez_x[1] << ' ' << l.perez_x[2] << ' ' << l.perez_x[3] << ' ' << l.perez_x[4] << ' ' << l.perez_x[5] << "], "
			  << "perez_Y["
			      << l.perez_y[0] << ' ' << l.perez_y[1] << ' ' << l.perez_y[2] << ' ' << l.perez_y[3] << ' ' << l.perez_y[4] << ' ' << l.perez_y[5] << "] "
			  << "]";
}


// TexMap
ostream& operator<< (ostream& os, const TexMap& t) {
	return os << "TexMap[" << t.rgbOffset << ", " << t.alphaOffset << ", " << t.width << "; " << t.height << "]";
}


}
