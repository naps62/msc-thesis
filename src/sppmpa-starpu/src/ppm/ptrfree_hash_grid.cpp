#include "ppm/ptrfree_hash_grid.h"
#include "ppm/kernels/helpers.cuh"

namespace ppm {

PtrFreeHashGrid :: PtrFreeHashGrid(const unsigned size, const unsigned hit_points_count) {
  this->size = size;
  hash_grid = NULL;
  lists = NULL;
  lengths = NULL;
  lists_index = NULL;

  this->lengths     = new unsigned int[hit_points_count];
  this->lists_index = new unsigned int[hit_points_count];
}

PtrFreeHashGrid :: ~PtrFreeHashGrid() {

}

}
