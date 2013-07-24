#include "ppm/ptrfree_hash_grid.h"
#include "ppm/kernels/helpers.cuh"

namespace ppm {

PtrFreeHashGrid :: PtrFreeHashGrid(const unsigned size) {
  this->size = size;
  hash_grid = NULL;
  lists = NULL;
  lengths = NULL;
  lists_index = NULL;
  //lengths_buff = NULL;
  //lists_buff = NULL;
  //index_buff = NULL;
}

PtrFreeHashGrid :: ~PtrFreeHashGrid() {

}

void PtrFreeHashGrid :: set_hit_points(std::vector<HitPointPosition>& hit_points_info, std::vector<HitPointRadiance>& hit_points) {
  this->hit_points_info = &hit_points_info[0];
  this->hit_points      = &hit_points[0];
  this->hit_points_count = hit_points_info.size();

  this->lengths     = new unsigned int[hit_points_count];
  this->lists_index = new unsigned int[hit_points_count];
}

void PtrFreeHashGrid :: set_bbox(BBox bbox) {
  this->bbox = bbox;
}


void PtrFreeHashGrid :: rehash() {
  const unsigned hit_points_count = this->hit_points_count;
  const BBox bbox = this->bbox;

  // calculate size of the grid cell
  float max_photon_radius = 0.f;
  for(unsigned i = 0; i < hit_points_count; ++i) {
    HitPointPosition& hpi = hit_points_info[i];
    HitPointRadiance& hp = hit_points[i];

    if (hpi.type == SURFACE)
      max_photon_radius = Max(max_photon_radius, hp.accum_photon_radius2);
  }

  const float cell_size = sqrtf(max_photon_radius) * 2.f;
  this->inv_cell_size = 1.f / cell_size;

  if (!hash_grid) {
    hash_grid = new std::deque<unsigned>*[size];
    for(unsigned i = 0; i < size; ++i)
      hash_grid[i] = NULL;
  }
  else {
    for(unsigned i = 0; i < size; ++i) {
      delete hash_grid[i];
      hash_grid[i] = NULL;
    }
  }

  unsigned long long entry_count = 0;
  for(unsigned i = 0; i < hit_points_count; ++i) {
    HitPointPosition& hpi = hit_points_info[i];

    if (hpi.type == SURFACE) {
      HitPointRadiance& hp = hit_points[i];
      const float photon_radius = sqrtf(hp.accum_photon_radius2);

      const Vector rad(photon_radius, photon_radius, photon_radius);
      const Vector b_min = ((hpi.position - rad) - bbox.pMin) * inv_cell_size;
      const Vector b_max = ((hpi.position + rad) - bbox.pMin) * inv_cell_size;


      for(int iz = abs(int(b_min.z)); iz <= abs(int(b_max.z)); ++iz) {
        for(int iy = abs(int(b_min.y)); iy <= abs(int(b_max.y)); ++iy) {
          for(int ix = abs(int(b_min.x)); ix <= abs(int(b_max.x)); ++ix) {
            int hv = kernels::helpers::hash(ix, iy, iz, size);

            if (hash_grid[hv] == NULL) {
              hash_grid[hv] = new std::deque<unsigned>();
            }

            hash_grid[hv]->push_front(i);
            entry_count++;
          }
        }
      }
    }
  }

  this->entry_count = entry_count;

  if (lists) delete[] lists;
  lists = new unsigned int[entry_count];

  uint list_index = 0;
  for(unsigned i = 0; i < hit_points_count; ++i) {
    std::deque<unsigned>* hps = hash_grid[i];
    lists_index[i] = list_index;

    if (hps) {
      lengths[i] = hps->size();
      for(std::deque<unsigned>::iterator iter = hps->begin(); iter != hps->end(); ++iter) {
        lists[list_index++] = *iter;
      }
    } else {
      lengths[i] = 0;
    }
  }
}

}
