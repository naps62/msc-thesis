#ifndef _PPM_PTRFREE_HASH_GRID_H_
#define _PPM_PTRFREE_HASH_GRID_H_

#include "ppm/types.h"
#include "luxrays/core.h"
#include <vector>

namespace ppm {

class PtrFreeHashGrid {
public:
  unsigned int* lists;
  unsigned int* lengths;
  unsigned int* lists_index;

  unsigned int size;
  float inv_cell_size;
  unsigned long long entry_count;

  PtrFreeHashGrid(const unsigned size, const unsigned hit_points_count);

  ~PtrFreeHashGrid();
};

}

#endif /* _PPM_PTRFREE_HASH_GRID_H_ */
