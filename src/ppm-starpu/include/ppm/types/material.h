#ifndef _PPM_TYPES_MATERIAL_H_
#define _PPM_TYPES_MATERIAL_H_

#include "utils/common.h"
#include "ppm/types.h"
#include <ostream>
using std::ostream;

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


struct AreaLightParam {
  Color gain;
};


struct MirrorParam {
  Color kr;
  int specular_bounce;
};


struct GlassParam {
  Color refl, refrct;
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
  Color kr;
  float exp;
  int specular_bounce;
};


struct MatteMetalParam {
  MatteParam matte;
  MetalParam metal;
  float matte_filter, tot_filter, matte_pdf, metal_pdf;
};


struct AlloyParam {
  Color diff, refl;
  float exp;
  float R0;
  int specular_bounce;
};


struct ArchGlassParam {
  Color refl, refrct;
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

ostream& operator<< (ostream& os, const MatteParam& m);
ostream& operator<< (ostream& os, const AreaLightParam& m);
ostream& operator<< (ostream& os, const MirrorParam& m);
ostream& operator<< (ostream& os, const GlassParam& m);
ostream& operator<< (ostream& os, const MatteMirrorParam& m);
ostream& operator<< (ostream& os, const MetalParam& m);
ostream& operator<< (ostream& os, const MatteMetalParam& m);
ostream& operator<< (ostream& os, const AlloyParam& m);
ostream& operator<< (ostream& os, const ArchGlassParam& m);
ostream& operator<< (ostream& os, const Material& m);

}

#endif // _PPM_TYPES_MATERIAL_H_
