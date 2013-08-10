#ifndef _PPM_PTRFREE_HASH_GRID_H_
#define _PPM_PTRFREE_HASH_GRID_H_

#include "ppm/types.h"
#include "luxrays/core.h"
#include <vector>
//#include "cuda_utils.h"

namespace ppm {

class PtrFreeHashGrid {
public:

  std::deque<unsigned> **hash_grid;

  unsigned int* lists;
  unsigned int* lengths;
  unsigned int* lists_index;

  unsigned int size;
  float inv_cell_size;
  unsigned int entry_count;

  //unsigned int* lists_buff;
  //unsigned int* lengths_buff;
  //unsigned int* index_buff;

  HitPointStaticInfo* hit_points_info;
  HitPoint*           hit_points;
  unsigned hit_points_count;

  BBox bbox;


  PtrFreeHashGrid(const unsigned size);

  ~PtrFreeHashGrid();

  void set_hit_points(std::vector<HitPointStaticInfo>& hit_points_info, std::vector<HitPoint>& hit_points);
  void set_bbox(BBox bbox);

  void rehash();

  //void AddFlux(PointerFreeScene *ss, const float alpha, const Point &hitPoint, const Normal &shadeN,
  //    const Vector wi, const Spectrum photonFlux,float currentPhotonRadius2);

  //void updateLookupTable();


};

}

#endif /* _PPM_PTRFREE_HASH_GRID_H_ */
