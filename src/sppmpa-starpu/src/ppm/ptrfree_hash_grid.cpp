#include "ppm/ptrfree_hash_grid.h"
#include "ppm/kernels/helpers.cuh"

namespace ppm {

PtrFreeHashGrid :: PtrFreeHashGrid(const unsigned size) {
  this->size = size;
  hash_grid = NULL;
  lists = NULL;
  lengths = NULL;
  lists_index = NULL;
}

PtrFreeHashGrid :: ~PtrFreeHashGrid() {

}

void PtrFreeHashGrid :: set_hit_points(std::vector<HitPointPosition>& hit_points_info, std::vector<HitPointRadiance>& hit_points) {
  this->hit_points_count = hit_points_info.size();

  this->lengths     = new unsigned int[hit_points_count];
  this->lists_index = new unsigned int[hit_points_count];
}

void PtrFreeHashGrid :: set_bbox(BBox bbox) {
  this->bbox = bbox;
}

}
