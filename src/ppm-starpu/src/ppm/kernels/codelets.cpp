#include "ppm/kernels/codelets.h"
#include "ppm/kernels/cpu_headers.h"

#include <limits>


namespace ppm { namespace kernels {

  namespace codelets {
    starpu_codelet generate_eye_paths;
    starpu_codelet intersect_ray_hit_buffer;
    starpu_codelet advance_eye_paths;

    starpu_perfmodel generate_eye_paths_pm;
    starpu_perfmodel intersect_ray_hit_buffer_pm;
    starpu_perfmodel advance_eye_paths_pm;

    const char* generate_eye_paths_sym       = "ppm_generate_eye_paths_001";
    const char* intersect_ray_hit_buffer_sym = "ppm_intersect_ray_hit_buffer_001";
    const char* advance_eye_paths_sym        = "ppm_advance_eye_paths_001";

    void perfmodel_init(starpu_perfmodel* model) {
      memset(model, 0, sizeof(starpu_perfmodel));
    }

    void init() {
      starpu_perfmodel* pm;
      starpu_codelet* cl;

      // generate_eye_paths
      pm = &generate_eye_paths_pm;
      pm->type   = STARPU_HISTORY_BASED;
      pm->symbol = generate_eye_paths_sym;

      cl   = &generate_eye_paths;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::generate_eye_paths;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_W;
      cl->modes[1]        = STARPU_RW;
      cl->model           = pm;


      // intersect_ray_hit_buffer
      pm = &intersect_ray_hit_buffer_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = intersect_ray_hit_buffer_sym;

      cl = &intersect_ray_hit_buffer;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::intersect_ray_hit_buffer;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_R;
      cl->modes[1]        = STARPU_RW;
      cl->model           = pm;

      // advance_eye_paths
      pm = &advance_eye_paths_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = advance_eye_paths_sym;

      cl = &advance_eye_paths;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::advance_eye_paths;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 5;
      cl->modes[0]        = STARPU_RW;
      cl->modes[1]        = STARPU_R;
      cl->modes[2]        = STARPU_RW;
      cl->modes[3]        = STARPU_R;
      cl->modes[4]        = STARPU_RW;
      cl->model           = pm;
    }
  }

} }
