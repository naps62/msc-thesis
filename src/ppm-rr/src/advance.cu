/*
 * advance.cu
 *
 *  Created on: Sep 13, 2012
 *      Author: rr
 */

#include "core.h"
#include "pointerfreescene.h"
#include "hitpoints.h"
#include "renderEngine.h"
#include "cuda_utils.h"
#include "stdio.h"
#include "my_cutil_math.h"
#include "CUDA_Worker.h"

// Using invDir0/invDir1/invDir2 and sign0/sign1/sign2 instead of an
// array because I dont' trust OpenCL compiler =)
__device__ int4 QBVHNode_BBoxIntersect(const float4 bboxes_minX, const float4 bboxes_maxX,
    const float4 bboxes_minY, const float4 bboxes_maxY, const float4 bboxes_minZ,
    const float4 bboxes_maxZ, const POINTERFREESCENE::QuadRay *ray4, const float4 invDir0,
    const float4 invDir1, const float4 invDir2, const int signs0, const int signs1,
    const int signs2) {
  float4 tMin = ray4->mint;
  float4 tMax = ray4->maxt;

  // X coordinate
  tMin = fmaxf(tMin, (bboxes_minX - ray4->ox) * invDir0);
  tMax = fminf(tMax, (bboxes_maxX - ray4->ox) * invDir0);

  // Y coordinate
  tMin = fmaxf(tMin, (bboxes_minY - ray4->oy) * invDir1);
  tMax = fminf(tMax, (bboxes_maxY - ray4->oy) * invDir1);

  // Z coordinate
  tMin = fmaxf(tMin, (bboxes_minZ - ray4->oz) * invDir2);
  tMax = fminf(tMax, (bboxes_maxZ - ray4->oz) * invDir2);

  // Return the visit flags
  return (tMax >= tMin);
}

__device__ void QuadTriangle_Intersect(const float4 origx, const float4 origy, const float4 origz,
    const float4 edge1x, const float4 edge1y, const float4 edge1z, const float4 edge2x,
    const float4 edge2y, const float4 edge2z, const uint4 primitives,
    POINTERFREESCENE::QuadRay *ray4, RayHit *rayHit) {
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

__global__ void Intersect(Ray *rays, RayHit *rayHits, POINTERFREESCENE::QBVHNode *nodes,
    POINTERFREESCENE::QuadTriangle *quadTris, const uint rayCount) {

  //  // Select the ray to check
  //  int len_X = gridDim.x * blockDim.x;
  //  int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
  //  int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
  //
  //  int gid = pos_y * len_X + pos_x;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < rayCount) {

    // Prepare the ray for intersection
    POINTERFREESCENE::QuadRay ray4;
    {
      float4 *basePtr = (float4 *) &rays[gid];
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

    RayHit rayHit;
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
        POINTERFREESCENE::QBVHNode *node = &nodes[nodeData];
        const int4 visit = QBVHNode_BBoxIntersect(node->bboxes[signs0][0],
            node->bboxes[1 - signs0][0], node->bboxes[signs1][1],
            node->bboxes[1 - signs1][1], node->bboxes[signs2][2],
            node->bboxes[1 - signs2][2], &ray4, invDir0, invDir1, invDir2, signs0,
            signs1, signs2);

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
          POINTERFREESCENE::QuadTriangle *quadTri = &quadTris[primNumber];
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
          QuadTriangle_Intersect(origx, origy, origz, edge1x, edge1y, edge1z, edge2x,
              edge2y, edge2z, primitives, &ray4, &rayHit);
        }
      }
    }

    //printf(\"MaxDepth=%02d\\n\", maxDepth);

    // Write result
    rayHits[gid].t = rayHit.t;
    rayHits[gid].b1 = rayHit.b1;
    rayHits[gid].b2 = rayHit.b2;
    rayHits[gid].index = rayHit.index;

    //printf("rayHits[%d].index = %u,t = %.4f\n",gid,rayHits[gid].index,rayHits[gid].t);

  }
}

__device__ void subIntersect(Ray& ray, POINTERFREESCENE::QBVHNode *nodes,
    POINTERFREESCENE::QuadTriangle *quadTris, RayHit& rayHit) {

  // Prepare the ray for intersection
  POINTERFREESCENE::QuadRay ray4;
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
      POINTERFREESCENE::QBVHNode *node = &nodes[nodeData];
      const int4 visit = QBVHNode_BBoxIntersect(node->bboxes[signs0][0],
          node->bboxes[1 - signs0][0], node->bboxes[signs1][1],
          node->bboxes[1 - signs1][1], node->bboxes[signs2][2],
          node->bboxes[1 - signs2][2], &ray4, invDir0, invDir1, invDir2, signs0, signs1,
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
        POINTERFREESCENE::QuadTriangle *quadTri = &quadTris[primNumber];
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

  //printf(\"MaxDepth=%02d\\n\", maxDepth);

  // Write result
  //    rayHit.t = rayHit.t;
  //    rayHit.b1 = rayHit.b1;
  //    rayHit.b2 = rayHit.b2;
  //    rayHits[gid].index = rayHit.index;
}

__device__ void InitPhotonPath(CUDA_Worker* w, PointerFreeScene* ss, PhotonPath& photonPath,
    Ray& ray, Seed& seed, unsigned long long& photonCount) {

  //Scene *scene = ss->scene;
  // Select one light source
  float lpdf;
  float pdf;

  Spectrum f;

  //photonPath->seed = mwc();

  float u0 = getFloatRNG(seed);
  float u1 = getFloatRNG(seed);
  float u2 = getFloatRNG(seed);
  float u3 = getFloatRNG(seed);
  float u4 = getFloatRNG(seed);
  float u5 = getFloatRNG(seed);

  int lightIndex;

  POINTERFREESCENE::LightSourceType lightT = ss->SampleAllLights(u0, &lpdf, lightIndex,
      w->infiniteLightBuff, w->sunLightBuff, w->skyLightBuff);

  if (lightT == POINTERFREESCENE::TYPE_IL_IS)
    ss->InfiniteLight_Sample_L(u1, u2, u3, u4, u5, &pdf, &ray, photonPath.flux,
        w->infiniteLightBuff, w->infiniteLightMapBuff);

  else if (lightT == POINTERFREESCENE::TYPE_SUN)
    ss->SunLight_Sample_L(u1, u2, u3, u4, u5, &pdf, &ray, photonPath.flux, w->sunLightBuff);

  else if (lightT == POINTERFREESCENE::TYPE_IL_SKY)
    ss->SkyLight_Sample_L(u1, u2, u3, u4, u5, &pdf, &ray, photonPath.flux, w->skyLightBuff);

  else {
    ss->TriangleLight_Sample_L(&w->areaLightsBuff[lightIndex], u1, u2, u3, u4, u5, &pdf, &ray,
        photonPath.flux, w->colorsBuff, w->meshDescsBuff);
  }

  //##########

  //const LightSource *light2 = scene->SampleAllLights(u0, &lpdf);

  // Initialize the photon path
  //photonPath->flux = light2->Sample_L(scene, u1, u2, u3, u4, u5, &pdf, ray);

  //#########


  photonPath.flux /= pdf * lpdf;
  photonPath.depth = 0;

  //engine->incPhotonCount();
  atomicAdd(&photonCount, 1);
  //printf("%u\n",atomicAdd(&photonCount, 1));
}

__device__ void AddFlux(CUDA_Worker* worker, PPM* engine, PointerFreeHashGrid* hashBuff,
    PointerFreeScene *ss, const float alpha, const Point &hitPoint, const Normal &shadeN,
    const Vector wi, const Spectrum photonFlux) {
  // Look for eye path hit points near the current hit point

  Vector hh = (hitPoint - hashBuff->hitPointsbbox.pMin) * hashBuff->invCellSize;

  const int ix = abs(int(hh.x));
      const int iy = abs(int(hh.y));
      const int iz = abs(int(hh.z));

      uint gridEntry = Hash(ix, iy, iz, hashBuff->hashGridSize);

      uint length = hashBuff->hashGridLenghtsBuff[gridEntry];
      if (length > 0) {

        uint localList = hashBuff->hashGridListsIndexBuff[gridEntry];

        for (uint i = localList; i < localList + length; i++) {

          //HitPointInfo *hp = &worker->hitPointsInfoBuff[worker->hashGridBuff[i]];
      HitPointStaticInfo *ihp = &worker->workerHitPointsInfoBuff[hashBuff->hashGridListsBuff[i]];
      HitPoint *hp = &worker->workerHitPointsBuff[hashBuff->hashGridListsBuff[i]];

      Vector v = ihp->position - hitPoint;
      // TODO: use configurable parameter for normal treshold
#if defined USE_SPPM || defined USE_PPM

      if ((Dot(ihp->normal, shadeN) > 0.5f)
          && (Dot(v, v) <= hp->accumPhotonRadius2)) {
#else
          if ((Dot(ihp->normal, shadeN) > 0.5f)
                  && (Dot(v, v) <= worker->currentPhotonRadius2)) {
#endif

        atomicAdd(&hp->accumPhotonCount, 1);

        Spectrum f;// = new Spectrum();
        // TODO mudar daqui para baixo conforme a alteração em cima. f deixou de ser um apontador, passou a ser uma struct

        POINTERFREESCENE::Material *hitPointMat = &worker->materialsBuff[ihp->materialSS];

        switch (hitPointMat->type) {

          case MAT_MATTE:
          ss->Matte_f(&hitPointMat->param.matte, ihp->wo, wi,
              shadeN, f);
          break;

          case MAT_MATTEMIRROR:
          ss->MatteMirror_f(&hitPointMat->param.matteMirror,
              ihp->wo, wi, shadeN, f);
          break;

          case MAT_MATTEMETAL:
          ss->MatteMetal_f(&hitPointMat->param.matteMetal, ihp->wo,
              wi, shadeN, f);
          break;

          case MAT_ALLOY:
          ss->Alloy_f(&hitPointMat->param.alloy, ihp->wo, wi,
              shadeN, f);

          break;
          default:

          break;

        }

        Spectrum flux = photonFlux * AbsDot(shadeN,
            wi) * ihp->throughput * f;

        //hp->accumReflectedFlux = (hp->accumReflectedFlux + flux) /** g*/;

        atomicAdd(&hp->accumReflectedFlux.r ,flux.r);
        atomicAdd(&hp->accumReflectedFlux.g ,flux.g);
        atomicAdd(&hp->accumReflectedFlux.b ,flux.b);

      }
    }
  }
}

__device__ bool GetHitPointInformation(CUDA_Worker* worker, PointerFreeScene *ss, Ray& ray,
    const RayHit& rayHit, Point &hitPoint, Spectrum &surfaceColor, Normal &N, Normal &shadeN) {

  hitPoint = (ray)(rayHit.t);
  const unsigned int currentTriangleIndex = rayHit.index;

  unsigned int currentMeshIndex;
  unsigned int triIndex;

  currentMeshIndex = worker->meshIDsBuff[currentTriangleIndex];
  triIndex = currentTriangleIndex - worker->meshFirstTriangleOffsetBuff[currentMeshIndex];

  POINTERFREESCENE::Mesh& m =
      ((POINTERFREESCENE::Mesh*) (worker->meshDescsBuff))[currentMeshIndex];

  if (m.hasColors) {

    ss->Mesh_InterpolateColor((Spectrum*) &worker->colorsBuff[m.colorsOffset],
        &worker->trisBuff[m.trisOffset], triIndex, rayHit.b1, rayHit.b2, &surfaceColor);

  } else {
    surfaceColor = Spectrum(1.f, 1.f, 1.f);
  }

  ss->Mesh_InterpolateNormal(&worker->normalsBuff[m.vertsOffset],
      &worker->trisBuff[m.trisOffset], triIndex, rayHit.b1, rayHit.b2, N);

  // Flip the normal if required
  if (Dot(ray.d, N) > 0.f)
    shadeN = -N;
  else
    shadeN = N;

  return false;
}

/**
 * threads will loop until all photons path traced. Need to merge with interscet.
 */
//__global__ void AdvancePhotonPath(CUDA_Worker* worker, PPM* engine,
//    CUDAScene *ss, PhotonPath *livePhotonPaths, unsigned long long* photonCount) {
//
//  //  int tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//  int len_X = gridDim.x * blockDim.x;
//  int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
//  int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
//
//  int tid = pos_y * len_X + pos_x;
//
//  if (tid >= worker->RayBufferSize)
//    return;
//
//  bool init = false;
//
//  PhotonPath *photonPath = &livePhotonPaths[tid];
//  Ray *ray = &worker->raysBuff[tid];
//  const RayHit *rayHit = &worker->hraysBuff[tid];
//
//  if (rayHit->Miss()) {
//    init = true;
//  } else { // Something was hit
//
//    Point hitPoint;
//    Spectrum surfaceColor;
//    Normal N, shadeN;
//
//    if (GetHitPointInformation(worker,ss, *ray, *rayHit, hitPoint, surfaceColor,
//        N, shadeN))
//      return;
//
//    const unsigned int currentTriangleIndex = rayHit->index;
//
//    const unsigned int currentMeshIndex =
//        worker->meshIDsBuff[currentTriangleIndex];
//
//    CUDASCENE::Material *hitPointMat =
//        &worker->materialsBuff[worker->meshMatsBuff[currentMeshIndex]];
//
//    uint matType = hitPointMat->type;
//
//    if (matType == MAT_AREALIGHT) {
//      init = true;
//    } else {
//      bool specularBounce;
//
//      float fPdf;
//      Vector wi;
//      Vector wo = -ray->d;
//
//      float u0 = getFloatRNG(worker->seedsBuff[tid]);
//      float u1 = getFloatRNG(worker->seedsBuff[tid]);
//      float u2 = getFloatRNG(worker->seedsBuff[tid]);
//
//      Spectrum f;
//
//      switch (matType) {
//case      MAT_MATTE:
//      ss->Matte_Sample_f(&hitPointMat->param.matte, &wo,
//          &wi, &fPdf, &f, &shadeN, u0, u1,&specularBounce);
//
//      f *= surfaceColor;
//      break;
//
//      case MAT_MIRROR:
//      ss->Mirror_Sample_f(&hitPointMat->param.mirror, &wo,
//          &wi, &fPdf, &f, &shadeN, &specularBounce);
//      f *= surfaceColor;
//      break;
//
//      case MAT_GLASS:
//      ss->Glass_Sample_f(&hitPointMat->param.glass, &wo,
//          &wi, &fPdf, &f, &N, &shadeN, u0, &specularBounce);
//      f *= surfaceColor;
//
//      break;
//
//      case MAT_MATTEMIRROR:
//      ss->MatteMirror_Sample_f(
//          &hitPointMat->param.matteMirror, &wo, &wi, &fPdf,
//          &f, &shadeN, u0, u1, u2, &specularBounce);
//      f *= surfaceColor;
//
//      break;
//
//      case MAT_METAL:
//      ss->Metal_Sample_f(&hitPointMat->param.metal, &wo,
//          &wi, &fPdf, &f, &shadeN, u0, u1, &specularBounce);
//      f *= surfaceColor;
//
//      break;
//
//      case MAT_MATTEMETAL:
//      ss->MatteMetal_Sample_f(
//          &hitPointMat->param.matteMetal, &wo, &wi, &fPdf,
//          &f, &shadeN, u0, u1, u2, &specularBounce);
//      f *= surfaceColor;
//
//      break;
//
//      case MAT_ALLOY:
//      ss->Alloy_Sample_f(&hitPointMat->param.alloy, &wo,
//          &wi, &fPdf, &f, &shadeN, u0, u1, u2,
//          &specularBounce);
//      f *= surfaceColor;
//
//      break;
//
//      case MAT_ARCHGLASS:
//      ss->ArchGlass_Sample_f(&hitPointMat->param.archGlass,
//          &wo, &wi, &fPdf, &f, &N, &shadeN, u0,
//          &specularBounce);
//      f *= surfaceColor;
//
//      break;
//
//      case MAT_NULL:
//      wi = ray->d;
//      specularBounce = 1;
//      fPdf = 1.f;
//      //printf("error\n");
//
//      break;
//
//      default:
//      // Huston, we have a problem...
//      //printf("error\n");
//
//      specularBounce = 1;
//      fPdf = 0.f;
//      break;
//    }
//
//    if (!specularBounce) { // if difuse
//      AddFlux(worker,engine,ss, engine->alpha, hitPoint, shadeN,
//          -ray->d, photonPath->flux, worker->invCellSize);
//
//    }
//
//    if (photonPath->depth < MAX_PHOTON_PATH_DEPTH) {
//      // Build the next vertex path ray
//      if ((fPdf <= 0.f) || f.Black()) {
//        init = true;
//      } else {
//        photonPath->depth++;
//        photonPath->flux *= f / fPdf;
//
//        // Russian Roulette
//        const float p = 0.75f;
//        if (photonPath->depth < 3) {
//          *ray = Ray(hitPoint, wi);
//        } else if (getFloatRNG(worker->seedsBuff[tid]) < p) {
//          photonPath->flux /= p;
//          *ray = Ray(hitPoint, wi);
//        } else {
//          init = true;
//        }
//      }
//    } else {
//      init = true;
//    }
//  }
//}
//
//if (init) {
//  InitPhotonPath(worker, ss, *photonPath, *ray,worker->seedsBuff[tid], *photonCount);
//}
//
//}

__device__ void subAdvancePhotonPath(CUDA_Worker* worker, PointerFreeHashGrid* hashBuff,
    PPM* engine, PointerFreeScene *ss, PhotonPath& photonPath, Ray& ray, RayHit& rayHit,
    Seed& seed, bool& init) {

  if (rayHit.Miss()) {
    init = true;
  } else { // Something was hit

    Point hitPoint;
    Spectrum surfaceColor;
    Normal N, shadeN;

    if (GetHitPointInformation(worker, ss, ray, rayHit, hitPoint, surfaceColor, N, shadeN))
      return;

    const unsigned int currentTriangleIndex = rayHit.index;

    const unsigned int currentMeshIndex = worker->meshIDsBuff[currentTriangleIndex];

    POINTERFREESCENE::Material *hitPointMat =
        &worker->materialsBuff[worker->meshMatsBuff[currentMeshIndex]];

    uint matType = hitPointMat->type;

    if (matType == MAT_AREALIGHT) {
      init = true;
    } else {
      bool specularBounce;

      float fPdf;
      Vector wi;
      Vector wo = -ray.d;

      float u0 = getFloatRNG(seed);
      float u1 = getFloatRNG(seed);
      float u2 = getFloatRNG(seed);

      Spectrum f;

      switch (matType) {
case      MAT_MATTE:
      ss->Matte_Sample_f(&hitPointMat->param.matte, &wo,
          &wi, &fPdf, &f, &shadeN, u0, u1,&specularBounce);

      f *= surfaceColor;
      break;

      case MAT_MIRROR:
      ss->Mirror_Sample_f(&hitPointMat->param.mirror, &wo,
          &wi, &fPdf, &f, &shadeN, &specularBounce);
      f *= surfaceColor;
      break;

      case MAT_GLASS:
      ss->Glass_Sample_f(&hitPointMat->param.glass, &wo,
          &wi, &fPdf, &f, &N, &shadeN, u0, &specularBounce);
      f *= surfaceColor;

      break;

      case MAT_MATTEMIRROR:
      ss->MatteMirror_Sample_f(
          &hitPointMat->param.matteMirror, &wo, &wi, &fPdf,
          &f, &shadeN, u0, u1, u2, &specularBounce);
      f *= surfaceColor;

      break;

      case MAT_METAL:
      ss->Metal_Sample_f(&hitPointMat->param.metal, &wo,
          &wi, &fPdf, &f, &shadeN, u0, u1, &specularBounce);
      f *= surfaceColor;

      break;

      case MAT_MATTEMETAL:
      ss->MatteMetal_Sample_f(
          &hitPointMat->param.matteMetal, &wo, &wi, &fPdf,
          &f, &shadeN, u0, u1, u2, &specularBounce);
      f *= surfaceColor;

      break;

      case MAT_ALLOY:
      ss->Alloy_Sample_f(&hitPointMat->param.alloy, &wo,
          &wi, &fPdf, &f, &shadeN, u0, u1, u2,
          &specularBounce);
      f *= surfaceColor;

      break;

      case MAT_ARCHGLASS:
      ss->ArchGlass_Sample_f(&hitPointMat->param.archGlass,
          &wo, &wi, &fPdf, &f, &N, &shadeN, u0,
          &specularBounce);
      f *= surfaceColor;

      break;

      case MAT_NULL:
      wi = ray.d;
      specularBounce = 1;
      fPdf = 1.f;
      //printf("error\n");

      break;

      default:
      // Huston, we have a problem...
      //printf("error\n");

      specularBounce = 1;
      fPdf = 0.f;
      break;
    }

    if (!specularBounce) { // if difuse
      AddFlux(worker,engine,hashBuff,ss, engine->alpha, hitPoint, shadeN,
          -ray.d, photonPath.flux);

    }

    if (photonPath.depth < MAX_PHOTON_PATH_DEPTH) {
      // Build the next vertex path ray
      if ((fPdf <= 0.f) || f.Black()) {
        init = true;
      } else {
        photonPath.depth++;
        photonPath.flux *= f / fPdf;

        // Russian Roulette
        const float p = 0.75f;
        if (photonPath.depth < 3) {
          ray = Ray(hitPoint, wi);
        } else {

#ifdef WARP_RR
          volatile __shared__ float u;
          if (LANE0) u =getFloatRNG(seed);
          //          printf("%d - %f\n", threadIdx.x, u);

          if (u < p) {
            photonPath.flux /= p;
            ray = Ray(hitPoint, wi);
          } else {
            init = true;
          }

#else
          if (getFloatRNG(seed) < p) {
            photonPath.flux /= p;
            ray = Ray(hitPoint, wi);
          } else {
            init = true;
          }
#endif

        }
      }
    } else {
      init = true;
    }
  }
}

}

///**
// * A thread per raybuffer/workbuffer entry, when path finished initializes another
// */
//__global__ void fullAdvance(CUDA_Worker* worker, PPM* engine, PointerFreeScene *ss,
//    PointerFreeHashGrid* hashBuff, PhotonPath *livePhotonPaths,
//    unsigned long long* photonCount, uint photonTarget) {
//
//  //Select the ray to check
//  //  int len_X = gridDim.x * blockDim.x;
//  //  int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
//  //  int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
//  //
//  //  int tid = pos_y * len_X + pos_x;
//
//  //  int TidX = threadIdx.x + blockIdx.x * blockDim.x;
//  //  int TidY = threadIdx.y + blockIdx.y * blockDim.y;
//  //  int TidZ = threadIdx.z + blockIdx.z * blockDim.z;
//  //
//  //  /* mapped to 1 dimension */
//  //  int tid = TidX + TidY * gridDim.x * blockDim.x + TidZ * gridDim.x * blockDim.x * gridDim.y
//  //      * blockDim.y;
//
//  int tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//  if (tid < worker->WORK_BUCKET_SIZE) {
//    //return;
//
//    __threadfence();
//
//    Ray& ray = worker->raysBuff[tid];
//    RayHit& rayHit = worker->hraysBuff[tid];
//    Seed& seed = worker->seedsBuff[tid];
//    PhotonPath& photonPath = livePhotonPaths[tid];
//
//    //InitPhotonPath(ss, photonPath, ray, seed, *photonCount);
//
//    bool init = true;
//
//    while (*photonCount < photonTarget) {
//
//      if (init) {
//        InitPhotonPath(worker, ss, photonPath, ray, seed, *photonCount);
//        init = false;
//      }
//
//      subIntersect(ray, worker->d_qbvhBuff, worker->d_qbvhTrisBuff, rayHit);
//
//      subAdvancePhotonPath(worker, hashBuff, engine, ss, photonPath, ray, rayHit,
//          worker->seedsBuff[tid], init);
//
//    }
//  }
//
//  //printf("%u \n",*photonCount);
//
//}


/**
 * Coherence tailored. Round robin
 */
__global__ void fullAdvance(CUDA_Worker* worker, PPM* engine, PointerFreeScene *ss,
    PointerFreeHashGrid* hashBuff, PhotonPath *livePhotonPaths,
    unsigned long long* photonCount, uint photonTarget) {
  // Select the ray to check


  //  int len_X = gridDim.x * blockDim.x;
  //  int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
  //  int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
  //
  //  int tid = pos_y * len_X + pos_x;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  //*photonCount=0;

  if (tid < worker->WORK_BUCKET_SIZE) {
    bool done = true;
    for (int it = tid; it < photonTarget; it += worker->WORK_BUCKET_SIZE) {

      Ray& ray = worker->raysBuff[tid];
      RayHit& rayHit = worker->hraysBuff[tid];
      Seed& seed = worker->seedsBuff[tid];
      PhotonPath& photonPath = livePhotonPaths[tid];

#ifdef WARP_RR

      if( tid >=32*10 && tid < 32*11) printf("%d - %u\n", tid, photonPath.depth);

      if (__all(done)) {
        InitPhotonPath(worker, ss, photonPath, ray, seed, *photonCount);
        done = false;
      }
#else
      InitPhotonPath(worker, ss, photonPath, ray, seed, *photonCount);
      done = false;
#endif

      while (!done) {

        subIntersect(ray, worker->d_qbvhBuff, worker->d_qbvhTrisBuff, rayHit);

        subAdvancePhotonPath(worker, hashBuff, engine, ss, photonPath, ray, rayHit,
            worker->seedsBuff[tid], done);
      }

    }
  }
}

void intersect_wrapper(Ray *rays, RayHit *rayHits, POINTERFREESCENE::QBVHNode *nodes,
    POINTERFREESCENE::QuadTriangle *quadTris, uint rayCount) {

//  int sqrtn = sqrt(rayCount);

  //dim3 blockDIM = dim3(16, 16);
  //dim3 gridDIM = dim3((sqrtn / blockDIM.x) + 1, (sqrtn / blockDIM.y) + 1);

  dim3 blockDIM = dim3(BLOCKSIZE);
  dim3 gridDIM = dim3((rayCount / blockDIM.x) + 1);

  Intersect<<<gridDIM,blockDIM>>>
  (rays, rayHits, nodes, quadTris, rayCount);

  checkCUDAError("");

}

unsigned long long AdvancePhotonPath_wrapper(CUDA_Worker* worker, PPM* engine, uint photonTarget) {

  double start = WallClockTime();

  //int sqrtn = sqrt(worker->WORK_BUCKET_SIZE);

  //dim3 blockDIM = dim3(16, 16);
  //dim3 gridDIM = dim3((sqrtn / blockDIM.x) + 1, (sqrtn / blockDIM.y) + 1);

  //dim3 blockDIM = dim3(512, 1);
  //dim3 gridDIM = dim3((worker->RayBufferSize / blockDIM.x) + 1, 1);

  dim3 blockDIM = dim3(BLOCKSIZE, 1);
  dim3 gridDIM = dim3(SM * FACTOR, 1);

  unsigned long long photonCount = 0;
  unsigned long long* photonCountBuff;

  __E(cudaMalloc((void**) (&photonCountBuff), sizeof(unsigned long long)));
  __E(cudaMemset(photonCountBuff, 0, sizeof(unsigned long long)));
  //cudaMemcpy(photonCountBuff, &photonCount, sizeof(uint), cudaMemcpyHostToDevice);

  PPM* engineBuff;
  __E(cudaMalloc((void**) (&engineBuff), sizeof(PPM)));
  __E(cudaMemcpy(engineBuff, engine, sizeof(PPM), cudaMemcpyHostToDevice));

  CUDA_Worker* workerBuff;
  __E(cudaMalloc((void**) (&workerBuff), sizeof(CUDA_Worker)));
  __E(cudaMemcpy(workerBuff, worker, sizeof(CUDA_Worker), cudaMemcpyHostToDevice));

  PointerFreeScene* ssBuff;
  __E(cudaMalloc((void**) (&ssBuff), sizeof(PointerFreeScene)));
  __E(cudaMemcpy(ssBuff, engine->ss, sizeof(PointerFreeScene), cudaMemcpyHostToDevice));

  PointerFreeHashGrid* hashBuff;
  __E(cudaMalloc((void**) (&hashBuff), sizeof(PointerFreeHashGrid)));
  __E(cudaMemcpy(hashBuff, worker->lookupA, sizeof(PointerFreeHashGrid), cudaMemcpyHostToDevice));

  if (!worker->livePhotonPathsBuff) {
    __E(
        cudaMalloc((void**) (&worker->livePhotonPathsBuff),
            worker->WORK_BUCKET_SIZE * sizeof(PhotonPath)));
  }

  __E(cudaMemset(worker->livePhotonPathsBuff, 0, worker->WORK_BUCKET_SIZE * sizeof(PhotonPath)));

  checkCUDAmemory("before launch");

  __E(cudaDeviceSynchronize());

//  double startTrace = WallClockTime();

  fullAdvance<<<gridDIM,blockDIM>>>
  (workerBuff, engineBuff, ssBuff, hashBuff,worker->livePhotonPathsBuff, photonCountBuff, photonTarget);

  checkCUDAError("");

  __E(
      cudaMemcpy(&photonCount, photonCountBuff, sizeof(unsigned long long),
          cudaMemcpyDeviceToHost));

//  float MPhotonsSec = photonCount / ((WallClockTime()-startTrace) * 1000000.f);

  //printf("\nRate: %.3f MPhotons/sec\n",MPhotonsSec);

  __E(cudaFree(photonCountBuff));
  __E(cudaFree(engineBuff));
  __E(cudaFree(ssBuff));
  __E(cudaFree(workerBuff));
  __E(cudaFree(hashBuff));

  checkCUDAError("");

  worker->profiler->addPhotonTracingTime(WallClockTime()-start);
  worker->profiler->addPhotonsTraced(photonCount);

  //printf("Advanced %lu, asked %u\n", photonCount, photonTarget);

  return photonCount;

}
