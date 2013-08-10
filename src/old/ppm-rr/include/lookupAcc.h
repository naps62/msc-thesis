/*
 * lookupAcc.h
 *
 *  Created on: Nov 9, 2012
 *      Author: rr
 */

#ifndef LOOKUPACC_H_
#define LOOKUPACC_H_

#include "renderEngine.h"
#include "core.h"
#include "cuda_utils.h"


enum lookupAccType {
  HASH_GRID
};

class lookupAcc {
public:
  lookupAcc();
  virtual ~lookupAcc();

  virtual void ReHash(float currentPhotonRadius2) =0;

  virtual void AddFlux(PointerFreeScene *ss, const float alpha, const Point &hitPoint,
      const Normal &shadeN, const Vector wi, const Spectrum photonFlux, float currentPhotonRadius2) =0;
};

class HashGridLookup {
public:

  std::list<unsigned int> **hashGrid;
  unsigned int hashGridSize;
  float invCellSize;
  unsigned int hashGridEntryCount;

  HitPointStaticInfo* workerHitPointsInfo;
  HitPoint* workerHitPoints;

  BBox hitPointsbbox;

  HashGridLookup(uint size) {


    hashGridSize = size;
    hashGrid = NULL;
    workerHitPointsInfo = NULL;

  }

  void setHitpoints(HitPointStaticInfo* d,HitPoint* workerHitPoints_) {
    workerHitPointsInfo = d;
    workerHitPoints = workerHitPoints_;
  }

  void setBBox(BBox d) {
    hitPointsbbox = d;
  }

  ~HashGridLookup();

  void ReHash(float currentPhotonRadius2);

  void AddFlux(PointerFreeScene *ss, const float alpha, const Point &hitPoint, const Normal &shadeN,
      const Vector wi, const Spectrum photonFlux, float currentPhotonRadius2);

private:
  unsigned int Hash(const int ix, const int iy, const int iz) {
    return (unsigned int) ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) % hashGridSize;
  }

};

class PointerFreeHashGrid {
public:

  std::list<unsigned int> **hashGrid;

  unsigned int* hashGridLists;
  unsigned int* hashGridLenghts;
  unsigned int* hashGridListsIndex;

  unsigned int hashGridSize;
  float invCellSize;
  unsigned int hashGridEntryCount;

  unsigned int* hashGridListsBuff;
  unsigned int* hashGridLenghtsBuff;
  unsigned int* hashGridListsIndexBuff;

  HitPointStaticInfo* workerHitPointsInfo;
  HitPoint* workerHitPoints;

  //IterationHitPoint* iterationHitPointsBuff;

  BBox hitPointsbbox;


  PointerFreeHashGrid(uint size) {

    hashGridSize = size;

    hashGrid = NULL;

    hashGridLenghts = NULL;
    hashGridLists = NULL;
    hashGridListsIndex = NULL;

    hashGridLenghtsBuff = NULL;
    hashGridListsBuff = NULL;
    hashGridListsIndexBuff = NULL;
  }

  ~PointerFreeHashGrid();

  void setHitpoints(HitPointStaticInfo* d,HitPoint* workerHitPoints_) {
    workerHitPointsInfo = d;
    workerHitPoints = workerHitPoints_;
  }

  void setBBox(BBox d) {
    hitPointsbbox = d;
  }

  void ReHash(float currentPhotonRadius2);

  void AddFlux(PointerFreeScene *ss, const float alpha, const Point &hitPoint, const Normal &shadeN,
      const Vector wi, const Spectrum photonFlux,float currentPhotonRadius2);

  void updateLookupTable();


private:
  unsigned int Hash(const int ix, const int iy, const int iz) {
    return (unsigned int) ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) % hashGridSize;
  }

};

#endif /* LOOKUPACC_H_ */
