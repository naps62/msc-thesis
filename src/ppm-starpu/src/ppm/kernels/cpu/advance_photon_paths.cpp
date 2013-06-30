#include "ppm/kernels/advance_photon_paths.h"

#include "utils/config.h"
#include "ppm/ptrfreescene.h"
#include "utils/random.h"
#include "ppm/types.h"
#include "ppm/kernels/helpers.cuh"
using ppm::PtrFreeScene;
using ppm::EyePath;

#include <starpu.h>
#include <cstdio>
#include <cstddef>

namespace ppm { namespace kernels { namespace cpu {

void advance_photon_paths_impl(
    Ray* const rays,                const unsigned rays_count,
    PhotonPath* const photon_paths, const unsigned photon_paths_count,
    Seed* const seed_buffer,        // const unsigned seed_buffer_count,
    const Config* config,
    const PtrFreeScene* scene) {

  unsigned todo_photon_count = photon_paths_count;

  while(todo_photon_count < 0) {

  }
}


void advance_photon_paths(void* buffers[], void* args_orig) {
  // cl_args
  const args_advance_photon_paths* args = (args_advance_photon_paths*) args_orig;
  const Config*       config = static_cast<const Config*>(args->config);
  const PtrFreeScene* scene  = static_cast<const PtrFreeScene*>(args->scene);

  // buffers
  // rays
  Ray* const rays           = reinterpret_cast<Ray* const>(STARPU_VECTOR_GET_PTR(buffers[0]));
  const unsigned rays_count = STARPU_VECTOR_GET_NX(buffers[0]);
  // photon_paths
  PhotonPath* const photon_paths = reinterpret_cast<PhotonPath* const>(STARPU_VECTOR_GET_PTR(buffers[1]));
  const unsigned photon_paths_count = STARPU_VECTOR_GET_NX(buffers[1]);
  // seeds
  Seed* const seed_buffer          = reinterpret_cast<Seed* const>(STARPU_VECTOR_GET_PTR(buffers[2]));
  //const unsigned seed_buffer_count = STARPU_VECTOR_GET_NX(buffers[2]);


  advance_photon_paths_impl(rays,         rays_count,
                            photon_paths, photon_paths_count,
                            seed_buffer, // seed_buffer_count,
                            config,
                            scene);


}

} } }
