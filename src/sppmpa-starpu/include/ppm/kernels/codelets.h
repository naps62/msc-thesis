#ifndef _PPM_KERNELS_CODELETS_H_
#define _PPM_KERNELS_CODELETS_H_

#include <starpu.h>
#include "utils/config.h"
#include "utils/cuda_config.cuh"
#include "ppm/ptrfreescene.h"
#include "ppm/ptrfree_hash_grid.h"

namespace ppm { namespace kernels {

  namespace codelets {
    struct starpu_args {
      const Config*          cpu_config;
      const PtrFreeScene*    cpu_scene;
      PtrFreeHashGrid* cpu_hash_grid;
      const CUDA::Config*    gpu_config;
      const PtrFreeScene*    gpu_scene;
      PtrFreeHashGrid* gpu_hash_grid;

      starpu_args() { };
    };

    extern struct starpu_codelet init_seeds;
    extern struct starpu_codelet generate_eye_paths;
    extern struct starpu_codelet advance_eye_paths;
    extern struct starpu_codelet bbox_compute;
    extern struct starpu_codelet rehash;
    extern struct starpu_codelet generate_photon_paths;
    extern struct starpu_codelet advance_photon_paths;
    extern struct starpu_codelet accum_flux;
    extern struct starpu_codelet update_sample_buffer;
    extern struct starpu_codelet splat_to_film;

    extern starpu_args    generic_args;

    void init(const Config* cpu_config,       const PtrFreeScene* cpu_scene, PtrFreeHashGrid* cpu_hash_grid,
              const CUDA::Config* gpu_config, const PtrFreeScene* gpu_scene, PtrFreeHashGrid* gpu_hash_grid);

  }


} }

#endif // _PPM_KERNELS_CODELETS_H_
