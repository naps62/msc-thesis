#include "ppm/kernels/codelets.h"
#include "ppm/kernels/cpu_headers.h"

namespace ppm { namespace kernels {

  namespace codelets {
    starpu_codelet generate_eye_paths;
    starpu_codelet intersect_ray_hit_buffer;

    void init() {
      // generate_eye_paths
      starpu_codelet* cl = &generate_eye_paths;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = INT_MAX;
      cl->cpu_funcs[0]    = ppm::kernels::cpu::generate_eye_paths;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_W;
      cl->modes[1]        = STARPU_RW;

      // eye_paths_to_hit_points
      cl = &intersect_ray_hit_buffer;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_SPMD;
      cl->max_parallelism = INT_MAX;
      cl->cpu_funcs[0]    = ppm::kernels::cpu::intersect_ray_hit_buffer;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_R;
      cl->modes[1]        = STARPU_RW;
    }
  }

} }
