/*
 * lookupAcc.cpp
 *
 *  Created on: Nov 9, 2012
 *      Author: rr
 */

#include "lookupAcc.h"
//#include "cppbench.h"

lookupAcc::lookupAcc() {
  // TODO Auto-generated constructor stub

}

lookupAcc::~lookupAcc() {
  // TODO Auto-generated destructor stub
}

void HashGridLookup::ReHash(float currentPhotonRadius2,HitPointRadianceFlux *workerHitPoints) {

  const unsigned int hitPointsCount = engine->hitPointTotal;
  const BBox &hpBBox = hitPointsbbox;

  // Calculate the size of the grid cell
#if defined USE_SPPMPA || defined USE_PPMPA
  float maxPhotonRadius2 = currentPhotonRadius2;
#else
  float maxPhotonRadius2 = 0.f;
  for (unsigned int i = 0; i < hitPointsCount; ++i) {
    HitPointPositionInfo *ihp = engine->GetHitPointInfo(i);
    HitPointRadianceFlux *hp = &workerHitPoints[i];

    if (ihp->type == SURFACE)
    maxPhotonRadius2 = Max(maxPhotonRadius2, hp->accumPhotonRadius2);
  }

#endif
  const float cellSize = sqrtf(maxPhotonRadius2) * 2.f;
  //std::cerr << "Hash grid cell size: " << cellSize << std::endl;
  invCellSize = 1.f / cellSize;

  // TODO: add a tunable parameter for hashgrid size
  //hashGridSize = hitPointsCount;
  if (!hashGrid) {
    hashGrid = new std::list<uint>*[hashGridSize];

    for (unsigned int i = 0; i < hashGridSize; ++i)
      hashGrid[i] = NULL;
  } else {
    for (unsigned int i = 0; i < hashGridSize; ++i) {
      delete hashGrid[i];
      hashGrid[i] = NULL;
    }
  }

  //std::cerr << "Building hit points hash grid:" << std::endl;
  //std::cerr << "  0k/" << hitPointsCount / 1000 << "k" << std::endl;
  //unsigned int maxPathCount = 0;
  double lastPrintTime = WallClockTime();
  unsigned long long entryCount = 0;

  for (unsigned int i = 0; i < hitPointsCount; ++i) {

    if (WallClockTime() - lastPrintTime > 2.0) {
      std::cerr << "  " << i / 1000 << "k/" << hitPointsCount / 1000 << "k" << std::endl;
      lastPrintTime = WallClockTime();
    }

    HitPointPositionInfo *hp = engine->GetHitPointInfo(i);
    //HitPointPositionInfo *hp = &workerHitPointsInfo[i];

    if (hp->type == SURFACE) {

#if defined USE_SPPMPA || defined USE_PPMPA

      const float photonRadius = sqrtf(currentPhotonRadius2);

#else
      HitPointRadianceFlux *hpp = &workerHitPoints[i];
      const float photonRadius = sqrtf(hpp->accumPhotonRadius2);

#endif
      const Vector rad(photonRadius, photonRadius, photonRadius);
      const Vector bMin = ((hp->position - rad) - hpBBox.pMin) * invCellSize;
      const Vector bMax = ((hp->position + rad) - hpBBox.pMin) * invCellSize;

      for (int iz = abs(int(bMin.z)); iz <= abs(int(bMax.z)); iz++) {
        for (int iy = abs(int(bMin.y)); iy <= abs(int(bMax.y)); iy++) {
          for (int ix = abs(int(bMin.x)); ix <= abs(int(bMax.x)); ix++) {

            int hv = Hash(ix, iy, iz);

            if (hashGrid[hv] == NULL)
              hashGrid[hv] = new std::list<uint>();

            hashGrid[hv]->push_front(i);
            ++entryCount;
          }
        }
      }
    }
  }

  hashGridEntryCount = entryCount;

  //std::cerr << "Max. hit points in a single hash grid entry: " << maxPathCount << std::endl;
  //std::cerr << "Total hash grid entry: " << entryCount << std::endl;
  //std::cerr << "Avg. hit points in a single hash grid entry: " << entryCount / hashGridSize
  //    << std::endl;

  //printf("Sizeof %d\n", sizeof(HitPoint*));

  // HashGrid debug code
  /*for (unsigned int i = 0; i < hashGridSize; ++i) {
   if (hashGrid[i]) {
   if (hashGrid[i]->size() > 10) {
   std::cerr << "HashGrid[" << i << "].size() = " <<hashGrid[i]->size() << std::endl;
   }
   }
   }*/
}

void HashGridLookup::AddFlux(HitPointRadianceFlux *workerHitPoints, PointerFreeScene *ss, const float alpha, const Point &hitPoint,
    const Normal &shadeN, const Vector wi, const Spectrum photonFlux,
    float currentPhotonRadius2) {


  // Look for eye path hit points near the current hit point
  Vector hh = (hitPoint - hitPointsbbox.pMin) * invCellSize;
  const int ix = abs(int(hh.x));
  const int iy = abs(int(hh.y));
  const int iz = abs(int(hh.z));

  //  std::list<uint> *hps = hashGrid[Hash(ix, iy, iz, hashGridSize)];
  //  if (hps) {
  //    std::list<uint>::iterator iter = hps->begin();
  //    while (iter != hps->end()) {
  //
  //      HitPoint *hp = &hitPoints[*iter++];


  uint gridEntry = Hash(ix, iy, iz);
  std::list<unsigned int>* hps = hashGrid[gridEntry];
  if (hps) {
    std::list<unsigned int>::iterator iter = hps->begin();
    while (iter != hps->end()) {
      HitPointPositionInfo *hp = engine->GetHitPointInfo(*iter);

      HitPointRadianceFlux *ihp = &workerHitPoints[*iter++];

      //Vector v = hp->position - hitPoint;

#if defined USE_SPPM || defined USE_PPM
      //if ((Dot(hp->normal, shadeN) > 0.5f) && (Dot(v, v) <= ihp->accumPhotonRadius2)) {

      const float dist2 = DistanceSquared(hp->position, hitPoint);
      if ((dist2 > ihp->accumPhotonRadius2))
      continue;

      const float dot = Dot(hp->normal, wi);
      if (dot <= 0.0001f)
      continue;

#else

      const float dist2 = DistanceSquared(hp->position, hitPoint);
      if ((dist2 > currentPhotonRadius2))
        continue;

      const float dot = Dot(hp->normal, wi);
      if (dot <= 0.0001f)
        continue;


#endif

      //const float g = (hp->accumPhotonCount * alpha + alpha)
      //    / (hp->accumPhotonCount * alpha + 1.f);


      //hp->photonRadius2 *= g;
      __sync_fetch_and_add(&ihp->accumPhotonCount, 1);

      Spectrum f;

      POINTERFREESCENE::Material *hitPointMat = &ss->materials[hp->materialSS];

      switch (hitPointMat->type) {

      case MAT_MATTE:
        ss->Matte_f(&hitPointMat->param.matte, hp->wo, wi, shadeN, f);
        break;

      case MAT_MATTEMIRROR:
        ss->MatteMirror_f(&hitPointMat->param.matteMirror, hp->wo, wi, shadeN, f);
        break;

      case MAT_MATTEMETAL:
        ss->MatteMetal_f(&hitPointMat->param.matteMetal, hp->wo, wi, shadeN, f);
        break;

      case MAT_ALLOY:
        ss->Alloy_f(&hitPointMat->param.alloy, hp->wo, wi, shadeN, f);
        break;
      default:
        break;

      }

      //f = hp->material->f(hp->wo, wi, shadeN);

      Spectrum flux = photonFlux * AbsDot(shadeN, wi) * hp->throughput * f;
      //if ((*iter-1) == 200) printf("%f\n", photonFlux.g);

#pragma omp critical
      {

        ihp->accumReflectedFlux = (ihp->accumReflectedFlux + flux) /** g*/;
      }
      //
      //        hp->accumReflectedFlux.r += flux.r;
      //        hp->accumReflectedFlux.r += flux.r;
      //        hp->accumReflectedFlux.r += flux.r;

      //printf("%f\n", f.r);


    }

  }
}

void PointerFreeHashGrid::ReHash(float currentPhotonRadius2,HitPointRadianceFlux *workerHitPoints) {

  const unsigned int hitPointsCount = engine->hitPointTotal;
  const BBox &hpBBox = hitPointsbbox;

  // Calculate the size of the grid cell
#if defined USE_SPPMPA || defined USE_PPMPA
  float maxPhotonRadius2 = currentPhotonRadius2;
#else
  float maxPhotonRadius2 = 0.f;
  for (unsigned int i = 0; i < hitPointsCount; ++i) {
    HitPointPositionInfo *ihp = engine->GetHitPointInfo(i);
    HitPointRadianceFlux *hp = &workerHitPoints[i];

    if (ihp->type == SURFACE)
    maxPhotonRadius2 = Max(maxPhotonRadius2, hp->accumPhotonRadius2);
  }

#endif
  const float cellSize = sqrtf(maxPhotonRadius2) * 2.f;
  //std::cerr << "Hash grid cell size: " << cellSize << std::endl;
  invCellSize = 1.f / cellSize;

  // TODO: add a tunable parameter for hashgrid size
  //hashGridSize = hitPointsCount;
  if (!hashGrid) {
    hashGrid = new std::list<uint>*[hashGridSize];

    for (unsigned int i = 0; i < hashGridSize; ++i)
      hashGrid[i] = NULL;
  } else {
    for (unsigned int i = 0; i < hashGridSize; ++i) {
      delete hashGrid[i];
      hashGrid[i] = NULL;
    }
  }

  //std::cerr << "Building hit points hash grid:" << std::endl;
  //std::cerr << "  0k/" << hitPointsCount / 1000 << "k" << std::endl;
  //unsigned int maxPathCount = 0;
  double lastPrintTime = WallClockTime();
  unsigned long long entryCount = 0;

  for (unsigned int i = 0; i < hitPointsCount; ++i) {

    if (WallClockTime() - lastPrintTime > 2.0) {
      std::cerr << "  " << i / 1000 << "k/" << hitPointsCount / 1000 << "k" << std::endl;
      lastPrintTime = WallClockTime();
    }

    HitPointPositionInfo *hp = engine->GetHitPointInfo(i);
    //HitPointPositionInfo *hp = &workerHitPointsInfo[i];

    if (hp->type == SURFACE) {
#if defined USE_SPPMPA || defined USE_PPMPA

      const float photonRadius = sqrtf(currentPhotonRadius2);

#else
      HitPointRadianceFlux *hpp = &workerHitPoints[i];
      const float photonRadius = sqrtf(hpp->accumPhotonRadius2);

#endif
      const Vector rad(photonRadius, photonRadius, photonRadius);
      const Vector bMin = ((hp->position - rad) - hpBBox.pMin) * invCellSize;
      const Vector bMax = ((hp->position + rad) - hpBBox.pMin) * invCellSize;

      for (int iz = abs(int(bMin.z)); iz <= abs(int(bMax.z)); iz++) {
        for (int iy = abs(int(bMin.y)); iy <= abs(int(bMax.y)); iy++) {
          for (int ix = abs(int(bMin.x)); ix <= abs(int(bMax.x)); ix++) {

            int hv = Hash(ix, iy, iz);

            //if (hv == engine->hitPointTotal - 1)

            if (hashGrid[hv] == NULL)
              hashGrid[hv] = new std::list<uint>();

            hashGrid[hv]->push_front(i);

            ++entryCount;

            /*// hashGrid[hv]->size() is very slow to execute
             if (hashGrid[hv]->size() > maxPathCount)
             maxPathCount = hashGrid[hv]->size();*/
          }
        }
      }
    }
  }

  hashGridEntryCount = entryCount;

  //std::cerr << "Max. hit points in a single hash grid entry: " << maxPathCount << std::endl;
  //std::cerr << "Total hash grid entry: " << entryCount << std::endl;
  //std::cerr << "Avg. hit points in a single hash grid entry: " << entryCount / hashGridSize
  //    << std::endl;

  //printf("Sizeof %d\n", sizeof(HitPoint*));

  // HashGrid debug code
  /*for (unsigned int i = 0; i < hashGridSize; ++i) {
   if (hashGrid[i]) {
   if (hashGrid[i]->size() > 10) {
   std::cerr << "HashGrid[" << i << "].size() = " <<hashGrid[i]->size() << std::endl;
   }
   }
   }*/
}

void PointerFreeHashGrid::updateLookupTable() {

//  __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Build Point Free lookup");


  if (hashGridLists)
    delete[] hashGridLists;

  hashGridLists = new uint[hashGridEntryCount];

  if (hashGridLenghts)
    memset(hashGridLenghts, 0, hashGridSize * sizeof(uint));
  else
    hashGridLenghts = new uint[hashGridSize];

  if (hashGridListsIndex)
    memset(hashGridListsIndex, 0, hashGridSize * sizeof(uint));
  else
    hashGridListsIndex = new uint[hashGridSize];

  uint listIndex = 0;
  for (unsigned int i = 0; i < hashGridSize; ++i) {

    std::list<uint> *hps = hashGrid[i];

    hashGridListsIndex[i] = listIndex;

    if (hps) {
      hashGridLenghts[i] = hps->size();
      std::list<uint>::iterator iter = hps->begin();
      while (iter != hps->end()) {
        hashGridLists[listIndex++] = *iter++;
      }
    } else {
      hashGridLenghts[i] = 0;
    }

  }

//  __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Build Point Free lookup");

//  __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Copy lookup to device");

  //checkCUDAmemory("before updateLookupTable");

  uint size1 = sizeof(uint) * hashGridEntryCount;

  if (hashGridListsBuff)
    cudaFree(hashGridListsBuff);
  cudaMalloc((void**) (&hashGridListsBuff), size1);

  cudaMemset(hashGridListsBuff, 0, size1);
  cudaMemcpy(hashGridListsBuff, hashGridLists, size1, cudaMemcpyHostToDevice);

  uint size2 = sizeof(uint) * hashGridSize;

  if (!hashGridListsIndexBuff)
    cudaMalloc((void**) (&hashGridListsIndexBuff), size2);

  cudaMemset(hashGridListsIndexBuff, 0, size2);

  cudaMemcpy(hashGridListsIndexBuff, hashGridListsIndex, size2, cudaMemcpyHostToDevice);

  if (!hashGridLenghtsBuff)
    cudaMalloc((void**) (&hashGridLenghtsBuff), size2);

  cudaMemset(hashGridLenghtsBuff, 0, size2);

  cudaMemcpy(hashGridLenghtsBuff, hashGridLenghts, size2, cudaMemcpyHostToDevice);

  checkCUDAError();

//  __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Copy lookup to device");


  //checkCUDAmemory("After updateLookupTable");

}
