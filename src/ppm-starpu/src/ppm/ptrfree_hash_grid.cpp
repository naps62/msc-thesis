#include "ppm/ptrfree_hash_grid.h"

PtrFreeHashGrid :: PtrFreeHashGrid(uint size) {
  this->size = size;
  grid = NULL;
  lengths = NULL;
  lists = NULL;
  lists_index = NULL;

  lengths_buff = NULL;
  lists_buff = NULL;
  lists_index_buff = NULL
}

PtrFreeHashGrid :: ~PtrFreeHashGrid() {

}

void PtrFreeHashGrid :: set_hit_points(HitPointStaticInfo* const hit_points, const unsigned hit_points_count) {
  this->hit_points = hit_points;
  this->hit_points_count = hit_points_count;
}

void PtrFreeHashGrid :: set_bbox(BBox bbox) {
  this->bbox = bbox;
}

unsigned PtrFreeHashGrid :: hash(const int ix, const int iy, const int iz) const {
  return (unsigned) ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) % size;
}


void PointerFreeHashGrid::ReHash(float /*currentPhotonRadius2*/) {

  const unsigned int hitPointsCount = engine->hitPointTotal;
  const BBox &hpBBox = hitPointsbbox;

  // Calculate the size of the grid cell
  float maxPhotonRadius2 = 0.f;
  for (unsigned int i = 0; i < hitPointsCount; ++i) {
    HitPointStaticInfo *ihp = &workerHitPointsInfo[i];
    HitPoint *hp = &workerHitPoints[i];

    if (ihp->type == SURFACE)
    maxPhotonRadius2 = Max(maxPhotonRadius2, hp->accumPhotonRadius2);
  }

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

    //HitPointInfo *hp = engine->GetHitPointInfo(i);
    HitPointStaticInfo *hp = &workerHitPointsInfo[i];

    if (hp->type == SURFACE) {
#if defined USE_SPPMPA || defined USE_PPMPA

      const float photonRadius = sqrtf(currentPhotonRadius2);

#else
      HitPoint *hpp = &workerHitPoints[i];
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
  std::cerr << "Total hash grid entry: " << entryCount << std::endl;
  std::cerr << "Avg. hit points in a single hash grid entry: " << entryCount
      / hashGridSize << std::endl;

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

