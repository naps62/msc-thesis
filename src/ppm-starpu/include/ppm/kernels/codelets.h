#ifndef _PPM_KERNELS_CODELETS_H_
#define _PPM_KERNELS_CODELETS_H_

#include <starpu.h>
#include "utils/config.h"
#include "utils/cuda_config.cuh"
#include "ppm/ptrfreescene.h"

namespace ppm { namespace kernels {

  namespace codelets {
    struct starpu_args {
      const Config*       cpu_config;
      const PtrFreeScene* cpu_scene;
      const CUDA::Config* gpu_config;
      const PtrFreeScene* gpu_scene;
    };

    extern starpu_codelet generate_eye_paths;
    extern starpu_codelet intersect_ray_hit_buffer;
    extern starpu_codelet advance_eye_paths;
    extern starpu_codelet generate_photon_paths;
    extern starpu_codelet advance_photon_paths;

    extern starpu_args    generic_args;

    void init(const Config* cpu_config, const PtrFreeScene* cpu_scene, const CUDA::Config* gpu_config, const PtrFreeScene* gpu_scene);

  }


} }

#endif // _PPM_KERNELS_CODELETS_H_
