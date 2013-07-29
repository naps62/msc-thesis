#include "ppm/kernels/codelets.h"
#include "ppm/kernels/helpers.cuh"
using namespace ppm::kernels::codelets;

#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cpu {


void rehash_impl(
    const HitPointPosition* const hit_points_info, unsigned size,
    PtrFreeHashGrid& hash_grid,
    unsigned long long* const entry_count,
    const BBox& bbox,
    const float current_photon_radius2) {

  const float cell_size = sqrtf(current_photon_radius2) * 2.f;
  const float inv_cell_size = 1.f / cell_size;

  if (!hash_grid.hash_grid) {
    hash_grid.hash_grid = new std::deque<unsigned>*[size];
    for(unsigned i = 0; i < size; ++i)
      hash_grid.hash_grid[i] = NULL;
  }
  else {
    for(unsigned i = 0; i < size; ++i) {
      delete hash_grid.hash_grid[i];
      hash_grid.hash_grid[i] = NULL;
    }
  }

  unsigned local_entry_count = 0;
  for(unsigned i = 0; i < size; ++i) {
    const HitPointPosition& hpi = hit_points_info[i];

    if (hpi.type == SURFACE) {
      const float photon_radius = sqrtf(current_photon_radius2);

      const Vector rad(photon_radius, photon_radius, photon_radius);
      const Vector b_min = ((hpi.position - rad) - bbox.pMin) * inv_cell_size;
      const Vector b_max = ((hpi.position + rad) - bbox.pMin) * inv_cell_size;

      for(int iz = abs(int(b_min.z)); iz <= abs(int(b_max.z)); ++iz) {
        for(int iy = abs(int(b_min.y)); iy <= abs(int(b_max.y)); ++iy) {
          for(int ix = abs(int(b_min.x)); ix <= abs(int(b_max.x)); ++ix) {
            int hv = helpers::hash(ix, iy, iz, size);

            if (hash_grid.hash_grid[hv] == NULL) {
              hash_grid.hash_grid[hv] = new std::deque<unsigned>();
            }

            hash_grid.hash_grid[hv]->push_front(i);
            local_entry_count++;
          }
        }
      }
    }
  }

  *entry_count = local_entry_count;

  if (hash_grid.lists) delete[] hash_grid.lists;
  hash_grid.lists = new unsigned int[local_entry_count];

  uint list_index = 0;
  for(unsigned i = 0; i < size; ++i) {
    std::deque<unsigned>* hps = hash_grid.hash_grid[i];
    hash_grid.lists_index[i] = list_index;

    if (hps) {
      hash_grid.lengths[i] = hps->size();
      for(std::deque<unsigned>::iterator iter = hps->begin(); iter != hps->end(); ++iter) {
        hash_grid.lists[list_index++] = *iter;
      }
    } else {
      hash_grid.lengths[i] = 0;
    }
  }
}


void rehash(void* buffers[], void* args_orig) {
  starpu_args args;
  BBox bbox;
  float current_photon_radius2;
  starpu_codelet_unpack_args(args_orig, &args, &bbox, &current_photon_radius2);

  const HitPointPosition* const hit_points_info = reinterpret_cast<const HitPointPosition* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned hit_points_count = STARPU_VECTOR_GET_NX(buffers[0]);

  unsigned long long* entry_count = reinterpret_cast<unsigned long long* const>(STARPU_VARIABLE_GET_PTR(buffers[1]));
  std::cout << current_photon_radius2 << '\n';
  rehash_impl(hit_points_info,
              hit_points_count,
              *(args.cpu_hash_grid),
              entry_count,
              bbox,
              current_photon_radius2);
}

} } }
