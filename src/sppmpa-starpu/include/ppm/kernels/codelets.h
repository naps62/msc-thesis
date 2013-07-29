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
      const PtrFreeHashGrid* cpu_hash_grid;
      const CUDA::Config*    gpu_config;
      const PtrFreeScene*    gpu_scene;
      const PtrFreeHashGrid* gpu_hash_grid;
    };

    struct starpu_advance_eye_paths_args {
      const PtrFreeScene* cpu_scene;
      const PtrFreeScene* gpu_scene;
      const Config*       cpu_config;
      const CUDA::Config* gpu_config;
    };

    struct starpu_advance_photon_paths_args {
      const Config*          cpu_config;
      const PtrFreeScene*    cpu_scene;
      const PtrFreeHashGrid* cpu_hash_grid;
      const CUDA::Config*    gpu_config;
      const PtrFreeScene*    gpu_scene;
      const PtrFreeHashGrid* gpu_hash_grid;
      const float photon_radius2;
    };

    struct starpu_accum_flux_args {
      const Config* cpu_config;
      const CUDA::Config* cuda_config;
      const unsigned photons_traced;
      const float current_photon_radius2;
    };

    extern starpu_codelet generate_eye_paths;
    extern starpu_codelet advance_eye_paths;
    extern starpu_codelet generate_photon_paths;
    extern starpu_codelet advance_photon_paths;
    extern starpu_codelet accum_flux;

    extern starpu_args    generic_args;

    void init(const Config* cpu_config,       const PtrFreeScene* cpu_scene, const PtrFreeHashGrid* cpu_hash_grid,
              const CUDA::Config* gpu_config, const PtrFreeScene* gpu_scene, const PtrFreeHashGrid* gpu_hash_grid);

  }


} }

#endif // _PPM_KERNELS_CODELETS_H_
