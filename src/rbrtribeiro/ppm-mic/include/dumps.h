#ifndef _PPM_DUMPS_H
#define _PPM_DUMPS_H

#include "pointerfreescene_types.h"
#include <ostream>
using std::ostream;

namespace POINTERFREESCENE {

// Mesh
ostream& operator<< (ostream& os, const Mesh& m);

// Material
ostream& operator<< (ostream& os, const Material& m);

// MatteParam
ostream& operator<< (ostream& os, const MatteParam& m);
// AreaLightParam
ostream& operator<< (ostream& os, const AreaLightParam& m);
// MirrorParam
ostream& operator<< (ostream& os, const MirrorParam& m);
// GlassParam
ostream& operator<< (ostream& os, const GlassParam& m);
// MatteMirrorParam
ostream& operator<< (ostream& os, const MatteMirrorParam& m);
// MetalParam
ostream& operator<< (ostream& os, const MetalParam& m);
// MatteMetalParam
ostream& operator<< (ostream& os, const MatteMetalParam& m);
// AlloyParam
ostream& operator<< (ostream& os, const AlloyParam& m);

// ArchGlassParam
ostream& operator<< (ostream& os, const ArchGlassParam& m);

// TriangleLight
ostream& operator<< (ostream& os, const TriangleLight& l);

// InfiniteLight
ostream& operator<< (ostream& os, const InfiniteLight& l);

// SunLight
ostream& operator<< (ostream& os, const SunLight& l);

// SkyLight
ostream& operator<< (ostream& os, const SkyLight& l);

// TexMap
ostream& operator<< (ostream& os, const TexMap& t);

}


#endif
