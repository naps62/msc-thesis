#include "ppm/kernels/helpers.cuh"
#include "ppm/ptrfree_hash_grid.h"
#include "ppm/math.h"
#include "ppm/kernels/cu_math.cuh"
#include <limits>
#include <cfloat>

#define QBVH_STACK_SIZE 24

namespace ppm { namespace kernels {

namespace helpers {

__device__ int4 QBVHNode_BBoxIntersect(const float4 bboxes_minX, const float4 bboxes_maxX,
    const float4 bboxes_minY, const float4 bboxes_maxY, const float4 bboxes_minZ,
    const float4 bboxes_maxZ, const ppm::QuadRay& ray4, const float4 invDir0,
    const float4 invDir1, const float4 invDir2, const int signs0, const int signs1,
    const int signs2) {

  float4 tMin = ray4.mint;
  float4 tMax = ray4.maxt;

  // X coordinate
  tMin = fmaxf(tMin, (bboxes_minX - ray4.ox) * invDir0);
  tMax = fminf(tMax, (bboxes_maxX - ray4.ox) * invDir0);

  // Y coordinate
  tMin = fmaxf(tMin, (bboxes_minY - ray4.oy) * invDir1);
  tMax = fminf(tMax, (bboxes_maxY - ray4.oy) * invDir1);

  // Z coordinate
  tMin = fmaxf(tMin, (bboxes_minZ - ray4.oz) * invDir2);
  tMax = fminf(tMax, (bboxes_maxZ - ray4.oz) * invDir2);

  // Return the visit flags
  return (tMax >= tMin);
}

__device__ void QuadTriangle_Intersect(const float4 origx, const float4 origy, const float4 origz,
    const float4 edge1x, const float4 edge1y, const float4 edge1z, const float4 edge2x,
    const float4 edge2y, const float4 edge2z, const uint4 primitives,
    ppm::QuadRay *ray4, RayHit *rayHit) {
  //--------------------------------------------------------------------------
  // Calc. b1 coordinate

  const float4 s1x = (ray4->dy * edge2z) - (ray4->dz * edge2y);
  const float4 s1y = (ray4->dz * edge2x) - (ray4->dx * edge2z);
  const float4 s1z = (ray4->dx * edge2y) - (ray4->dy * edge2x);

  const float4 divisor = (s1x * edge1x) + (s1y * edge1y) + (s1z * edge1z);

  const float4 dx = ray4->ox - origx;
  const float4 dy = ray4->oy - origy;
  const float4 dz = ray4->oz - origz;

  const float4 b1 = ((dx * s1x) + (dy * s1y) + (dz * s1z)) / divisor;

  //--------------------------------------------------------------------------
  // Calc. b2 coordinate

  const float4 s2x = (dy * edge1z) - (dz * edge1y);
  const float4 s2y = (dz * edge1x) - (dx * edge1z);
  const float4 s2z = (dx * edge1y) - (dy * edge1x);

  const float4 b2 = ((ray4->dx * s2x) + (ray4->dy * s2y) + (ray4->dz * s2z)) / divisor;

  //--------------------------------------------------------------------------
  // Calc. b0 coordinate

  const float4 b0 = (make_float4(1.f)) - b1 - b2;

  //--------------------------------------------------------------------------

  const float4 t = ((edge2x * s2x) + (edge2y * s2y) + (edge2z * s2z)) / divisor;

  float _b1, _b2;
  float maxt = ray4->maxt.x;
  uint index;

  int4 cond = (divisor != make_float4(0.f)) & (b0 >= make_float4(0.f)) & (b1 >= make_float4(0.f))
      & (b2 >= make_float4(0.f)) & (t > ray4->mint);

  const int cond0 = cond.x && (t.x < maxt);
  maxt = select(maxt, t.x, cond0);
  _b1 = select(0.f, b1.x, cond0);
  _b2 = select(0.f, b2.x, cond0);
  index = select(0xffffffffu, primitives.x, cond0);

  const int cond1 = cond.y && (t.y < maxt);
  maxt = select(maxt, t.y, cond1);
  _b1 = select(_b1, b1.y, cond1);
  _b2 = select(_b2, b2.y, cond1);
  index = select(index, primitives.y, cond1);

  const int cond2 = cond.z && (t.z < maxt);
  maxt = select(maxt, t.z, cond2);
  _b1 = select(_b1, b1.z, cond2);
  _b2 = select(_b2, b2.z, cond2);
  index = select(index, primitives.z, cond2);

  const int cond3 = cond.w && (t.w < maxt);
  maxt = select(maxt, t.w, cond3);
  _b1 = select(_b1, b1.w, cond3);
  _b2 = select(_b2, b2.w, cond3);
  index = select(index, primitives.w, cond3);

  if (index == 0xffffffffu)
    return;

  ray4->maxt = make_float4(maxt);

  rayHit->t = maxt;
  rayHit->b1 = _b1;
  rayHit->b2 = _b2;
  rayHit->index = index;
}

__device__ void subIntersect(Ray& ray, ppm::QBVHNode *nodes,
    ppm::QuadTriangle *quadTris, RayHit& rayHit) {

  // Prepare the ray for intersection
  ppm::QuadRay ray4;
  {
    float4 *basePtr = (float4 *) &ray;
    float4 data0 = (*basePtr++);
    float4 data1 = (*basePtr);

    ray4.ox = make_float4(data0.x);
    ray4.oy = make_float4(data0.y);
    ray4.oz = make_float4(data0.z);

    ray4.dx = make_float4(data0.w);
    ray4.dy = make_float4(data1.x);
    ray4.dz = make_float4(data1.y);

    ray4.mint = make_float4(data1.z);
    ray4.maxt = make_float4(data1.w);
  }

  const float4 invDir0 = make_float4(1.f / ray4.dx.x);
  const float4 invDir1 = make_float4(1.f / ray4.dy.x);
  const float4 invDir2 = make_float4(1.f / ray4.dz.x);

  const int signs0 = (ray4.dx.x < 0.f);
  const int signs1 = (ray4.dy.x < 0.f);
  const int signs2 = (ray4.dz.x < 0.f);

  //RayHit rayHit;
  rayHit.index = 0xffffffffu;

  int nodeStack[QBVH_STACK_SIZE];
  nodeStack[0] = 0; // first node to handle: root node

  //------------------------------
  // Main loop
  int todoNode = 0; // the index in the stack
  // nodeStack leads to a lot of local memory banks conflicts however it has not real
  // impact on performances (I guess access latency is hiden by other stuff).
  // Avoiding conflicts is easy to do but it requires to know the work group
  // size (not worth doing if there are not performance benefits).
  //__shared__ int *nodeStack = &nodeStacks[QBVH_STACK_SIZE * threadIdx.x];
  //nodeStack[0] = 0; // first node to handle: root node


  //int maxDepth = 0;
  while (todoNode >= 0) {
    const int nodeData = nodeStack[todoNode];
    --todoNode;

    // Leaves are identified by a negative index
    if (!QBVHNode_IsLeaf(nodeData)) {
      ppm::QBVHNode *node = &nodes[nodeData];
      //printf("%x\n", ray4);
      const int4 visit = QBVHNode_BBoxIntersect(node->bboxes[signs0][0],
          node->bboxes[1 - signs0][0], node->bboxes[signs1][1],
          node->bboxes[1 - signs1][1], node->bboxes[signs2][2],
          node->bboxes[1 - signs2][2], ray4, invDir0, invDir1, invDir2, signs0, signs1,
          signs2);


      const int4 children = node->children;

      // For some reason doing logic operations with int4 is very slow
      nodeStack[todoNode + 1] = children.w;
      todoNode += (visit.w && !QBVHNode_IsEmpty(children.w)) ? 1 : 0;
      nodeStack[todoNode + 1] = children.z;
      todoNode += (visit.z && !QBVHNode_IsEmpty(children.z)) ? 1 : 0;
      nodeStack[todoNode + 1] = children.y;
      todoNode += (visit.y && !QBVHNode_IsEmpty(children.y)) ? 1 : 0;
      nodeStack[todoNode + 1] = children.x;
      todoNode += (visit.x && !QBVHNode_IsEmpty(children.x)) ? 1 : 0;

      //maxDepth = max(maxDepth, todoNode);
    } else {
      // Perform intersection
      const uint nbQuadPrimitives = QBVHNode_NbQuadPrimitives(nodeData);
      const uint offset = QBVHNode_FirstQuadIndex(nodeData);

      for (uint primNumber = offset; primNumber < (offset + nbQuadPrimitives); ++primNumber) {
        ppm::QuadTriangle *quadTri = &quadTris[primNumber];
        const float4 origx = quadTri->origx;
        const float4 origy = quadTri->origy;
        const float4 origz = quadTri->origz;
        const float4 edge1x = quadTri->edge1x;
        const float4 edge1y = quadTri->edge1y;
        const float4 edge1z = quadTri->edge1z;
        const float4 edge2x = quadTri->edge2x;
        const float4 edge2y = quadTri->edge2y;
        const float4 edge2z = quadTri->edge2z;
        const uint4 primitives = quadTri->primitives;
        QuadTriangle_Intersect(origx, origy, origz, edge1x, edge1y, edge1z, edge2x, edge2y,
            edge2z, primitives, &ray4, &rayHit);
      }
    }
  }
}

}

} }
