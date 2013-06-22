#include "ppm/ptrfree_hash_grid.h"

namespace ppm {

PtrFreeHashGrid :: PtrFreeHashGrid() {
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

void PtrFreeHashGrid :: set_hit_points(std::vector<HitPointStaticInfo> hit_points_info, vector<HitPoint> hit_points) {
  this->hit_points_info = &hit_points_info[0];
  this->hit_points      = &hit_points[0];
  this->hit_points_count = hit_points_info.size();
}

void PtrFreeHashGrid :: set_bbox(BBox bbox) {
  this->bbox = bbox;
}

unsigned PtrFreeHashGrid :: hash(const int ix, const int iy, const int iz) const {
  return (unsigned) ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) % size;
}


void PtrFreeHashGrid :: rehash() {
  const unsigned hit_points_count = this->hit_points_count;
  const BBox bbox = this->bbox;

  // calculate size of the grid cell
  float max_photon_radius = 0.f;
  for(unsigned i = 0; i < hit_points_count; ++i) {
    HitPointStaticInfo& hpi = hit_points_info[i];

    if (hpi->type == SURFACE)
      max_photon_radius = Max(max_photon_radius, hpi.accum_photon_radius);
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
    HitPointStaticInfo& hpi = hit_points_info[i];

    if (hpi->type == SURFACE) {
      HitPoint& hp = hit_points[i];
      const float photon_radius = sqrtf(hp.accum_photon_radius);

      const Vector rad(photon_radius, photon_radius, photon_radius);
      const Vector b_min = ((hp.position - rad) - bbox.p_min) * inv_cell_size;
      const Vector b_max = ((hp.position + rad) - bbox.p_min) * inv_cell_size;

      for(int iz = abs(int(b_min.z)); iz < abs(int(b_max.z)); ++iz) {
        for(int iy = abs(int(b_min.y)); iz < abs(int(b_max.y)); ++iy) {
          for(int ix = abs(int(b_min.x)); iz < abs(int(b_max.x)); ++ix) {
            int hv = this->hash(ix, iy, iz);

            if (hash_grid[hv] == NULL)
              hash_grid[hv] = new std::deque<unsigned>();

            hash_grid[hv].push_front(i);
            entry_count++;
          }
        }
      }
    }
  }

  this->entry_count = entry_count;
}

}
