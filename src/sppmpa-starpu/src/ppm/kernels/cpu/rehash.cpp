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
    unsigned*  hash_grid,
    unsigned*  hash_grid_lengths,
    unsigned*  hash_grid_indexes,
    float* inv_cell_size,
    const BBox& bbox,
    const float current_photon_radius2) {

  const float cell_size = sqrtf(current_photon_radius2) * 2.f;
  *inv_cell_size = 1.f / cell_size;

  std::deque<unsigned>* hash_grid_deque[size];
  for(unsigned i = 0; i < size; ++i) {
    hash_grid_deque[i] = NULL;
  }


  // unsigned entry_count = 0;
  for(unsigned i = 0; i < size; ++i) {
    const HitPointPosition& hpi = hit_points_info[i];

    if (hpi.type == SURFACE) {
      const float photon_radius = sqrtf(current_photon_radius2);

      const Vector rad(photon_radius, photon_radius, photon_radius);
      const Vector b_min = ((hpi.position - rad) - bbox.pMin) * (*inv_cell_size);
      const Vector b_max = ((hpi.position + rad) - bbox.pMin) * (*inv_cell_size);

      for(int iz = abs(int(b_min.z)); iz <= abs(int(b_max.z)); ++iz) {
        for(int iy = abs(int(b_min.y)); iy <= abs(int(b_max.y)); ++iy) {
          for(int ix = abs(int(b_min.x)); ix <= abs(int(b_max.x)); ++ix) {
            int hv = helpers::hash(ix, iy, iz, size);

            if (hash_grid_deque[hv] == NULL) {
              hash_grid_deque[hv] = new std::deque<unsigned>();
            }

            hash_grid_deque[hv]->push_front(i);
            // entry_count++;
          }
        }
      }
    }
  }

  uint list_index = 0;
  for(unsigned i = 0; i < size; ++i) {
    std::deque<unsigned>* hps = hash_grid_deque[i];
    hash_grid_indexes[i] = list_index;

    if (hps) {
      hash_grid_lengths[i] = hps->size();
      for(std::deque<unsigned>::iterator iter = hps->begin(); iter != hps->end(); ++iter) {
        assert(list_index < size*8);
        hash_grid[list_index++] = *iter;
      }
    } else {
      hash_grid_lengths[i] = 0;
    }
  }
}


void rehash(void* buffers[], void* args_orig) {
  starpu_args args;
  starpu_codelet_unpack_args(args_orig, &args);


  const HitPointPosition* const hit_points_info = reinterpret_cast<const HitPointPosition* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned hit_points_count = STARPU_VECTOR_GET_NX(buffers[0]);

  const BBox* const bbox = (const BBox* const)STARPU_VARIABLE_GET_PTR(buffers[1]);
  const float* const current_photon_radius2 = (const float* const)STARPU_VARIABLE_GET_PTR(buffers[2]);

  unsigned*           hash_grid      = (unsigned*) STARPU_VECTOR_GET_PTR(buffers[3]);
  unsigned*           lengths        = (unsigned*) STARPU_VECTOR_GET_PTR(buffers[4]);
  unsigned*           indexes        = (unsigned*) STARPU_VECTOR_GET_PTR(buffers[5]);
  float*              inv_cell_size  = (float*)    STARPU_VARIABLE_GET_PTR(buffers[6]);

  rehash_impl(hit_points_info,
              hit_points_count,
              hash_grid,
              lengths,
              indexes,
              inv_cell_size,
              *bbox,
              *current_photon_radius2);

}

} } }
