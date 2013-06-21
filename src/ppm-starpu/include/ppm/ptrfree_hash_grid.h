#ifndef _PPM_PTRFREE_HASH_GRID_H_
#define _PPM_PTRFREE_HASH_GRID_H_

//#include "renderEngine.h"
#include "ppm/core.h"
//#include "cuda_utils.h"

namespace ppm {

class PtrFreeHashGrid {
public:

  std::list<unsigned int> **hash_grid;

  unsigned int* lists;
  unsigned int* lenghts;
  unsigned int* lists_index;

  unsigned int size;
  float inv_cell_size;
  unsigned int entry_count;

  unsigned int* Lists_buff;
  unsigned int* lenghts_buff;
  unsigned int* index_buff;

  HitPointStaticInfo* hit_points;
  unsigned hit_points_count;

  BBox bbox;


  PtrFreeHashGrid(uint size);

  ~PtrFreeHashGrid();

  void set_hit_points(HitPointStaticInfo* const hit_points, const unsigned hit_points_count);
  void set_bbox(BBox bbox);

  void rehash() {
    const unsigned hit_points_count = this->hit_points_count;
    const BBox bbox = this->bbox;

    // calculate size of the grid cell
    float max_photon_radius = 0.f;
    for(unsigned i = 0; i < hit_points_count; ++i) {
      HitPointStaticInfo& hp = hit_points[i];

      if (hp->type == SURFACE)
        max_photon_radius = Max(max_photon_radius, hp.accum_photon_radius);
    }

    const float cell_size = sqrtf(max_photon_radius) * 2.f;
    this->inv_cell_size = 1.f / cell_size;

    if (!hash_grid) {
      hash_grid = new std::list<unsigned>*[size];
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
      HitPointStaticInfo& hp = hit_points[i];

      if (hp->type == SURFACE) {

      }
    }
  }

  //void AddFlux(PointerFreeScene *ss, const float alpha, const Point &hitPoint, const Normal &shadeN,
  //    const Vector wi, const Spectrum photonFlux,float currentPhotonRadius2);

  //void updateLookupTable();

#endif

private:
  unsigned hash(const int ix, const int iy, const int iz) const;

};

}

#endif /* _PPM_PTRFREE_HASH_GRID_H_ */
