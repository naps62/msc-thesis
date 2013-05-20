#ifndef _PPM_KERNELS_CODELETS_H_
#define _PPM_KERNELS_CODELETES_H_

#include <starpu.h>

namespace ppm { namespace kernels {

  namespace codelets {

    extern starpu_codelet generate_eye_paths;
    extern starpu_codelet eye_paths_to_hit_points;

    void init();
  }


} }

#endif // _PPM_KERNELS_COMMON_H_
