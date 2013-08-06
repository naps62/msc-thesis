#ifndef _PPM_TYPES_H_
#define _PPM_TYPES_H_

#include "utils/common.h"
#include "luxrays/core/accelerator.h"
#include <ostream>
using std::ostream;

namespace ppm {

typedef enum {
  ACCEL_BVH   = luxrays::ACCEL_BVH,
  ACCEL_QBVH  = luxrays::ACCEL_QBVH,
  ACCEL_MQBVH = luxrays::ACCEL_MQBVH
} AcceleratorType;

typedef struct QuadRay {
  float4 ox, oy, oz;
  float4 dx, dy, dz;
  float4 mint, maxt;
} QuadRay;

typedef struct {
  float4 origx, origy, origz;
  float4 edge1x, edge1y, edge1z;
  float4 edge2x, edge2y, edge2z;
  uint4 primitives;
} QuadTriangle;

typedef struct {
  float4 bboxes[2][3];
  int4 children;
} QBVHNode;

#define _emptyLeafNode 0xffffffff

#define QBVHNode_IsLeaf(index) (index < 0)
#define QBVHNode_IsEmpty(index) (index == _emptyLeafNode)
#define QBVHNode_NbQuadPrimitives(index) ((uint)(((index >> 27) & 0xf) + 1))
#define QBVHNode_FirstQuadIndex(index) (index & 0x07ffffff)

struct Color {
  float r, g, b;
};

ostream& operator<< (ostream& os, const Color& c);


#define PPM_NONE 0xffffffffu

}

//#include "ppm/geometry/vector.h"
//#include "ppm/geometry/normal.h"
//#include "ppm/geometry/uv.h"
////#include "ppm/geometry/vector_normal.h"
//#include "ppm/geometry/point.h"
//#include "ppm/geometry/ray.h"
//#include "ppm/geometry/rayhit.h"
//#include "ppm/geometry/raybuffer.h"
//#include "ppm/geometry/bbox.h"
//#include "ppm/geometry/bsphere.h"
//#include "ppm/geometry/triangle.h"
//#include "ppm/geometry/matrix4x4.h"
//#include "ppm/geometry/mesh.h"
#include "luxrays/core.h"

#include "ppm/types/camera.h"
#include "ppm/types/material.h"
#include "ppm/types/light.h"
#include "ppm/types/texture.h"
#include "ppm/types/paths.h"
#include "ppm/types/hitpoints.h"
#include "ppm/types/mesh.h"

#endif // _PPM_TYPES_H_
