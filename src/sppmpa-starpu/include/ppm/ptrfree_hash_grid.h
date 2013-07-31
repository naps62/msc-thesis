#ifndef _PPM_PTRFREE_HASH_GRID_H_
#define _PPM_PTRFREE_HASH_GRID_H_

#include "ppm/types.h"
#include "luxrays/core.h"
#include <vector>

namespace ppm {

class PtrFreeHashGrid {
public:

  std::deque<unsigned> **hash_grid;

  unsigned int* lists;
  unsigned int* lengths;
  unsigned int* lists_index;

  unsigned int size;
  float inv_cell_size;
  unsigned long long entry_count;

  HitPointPosition* hit_points_info;
  HitPointRadiance* hit_points;
  unsigned hit_points_count;

  BBox bbox;

  PtrFreeHashGrid(const unsigned size);

  ~PtrFreeHashGrid();

  void set_hit_points(std::vector<HitPointPosition>& hit_points_info, std::vector<HitPointRadiance>& hit_points);
  void set_bbox(BBox bbox);
};

}

#endif /* _PPM_PTRFREE_HASH_GRID_H_ */
