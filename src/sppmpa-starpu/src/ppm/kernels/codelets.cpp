#include "ppm/kernels/codelets.h"
#include "ppm/kernels/cpu_headers.h"

#include <limits>


namespace ppm { namespace kernels {

  namespace codelets {
    starpu_args    generic_args;

    struct starpu_codelet generate_eye_paths;
    struct starpu_codelet advance_eye_paths;
    struct starpu_codelet bbox_compute;
    struct starpu_codelet bbox_zero_initialize;
    struct starpu_codelet bbox_reduce;
    struct starpu_codelet compute_photon_radius;
    struct starpu_codelet rehash;
    struct starpu_codelet generate_photon_paths;
    struct starpu_codelet advance_photon_paths;
    struct starpu_codelet accum_flux;

    starpu_perfmodel generate_eye_paths_pm;
    starpu_perfmodel advance_eye_paths_pm;
    starpu_perfmodel bbox_compute_pm;
    starpu_perfmodel bbox_zero_initialize_pm;
    starpu_perfmodel bbox_reduce_pm;
    starpu_perfmodel compute_photon_radius_pm;
    starpu_perfmodel rehash_pm;
    starpu_perfmodel generate_photon_paths_pm;
    starpu_perfmodel advance_photon_paths_pm;
    starpu_perfmodel accum_flux_pm;

    const char* generate_eye_paths_sym        = "ppm_generate_eye_paths_001";
    const char* advance_eye_paths_sym         = "ppm_advance_eye_paths_001";
    const char* bbox_compute_sym              = "ppm_bbox_compute_001";
    const char* bbox_zero_initialize_sym      = "ppm_bbox_zero_initialize_001";
    const char* bbox_reduce_sym               = "ppm_bbox_reduce_001";
    const char* compute_photon_radius_sym     = "ppm_bbox_reduce_001";
    const char* rehash_sym                    = "ppm_rehash_001";
    const char* generate_photon_paths_sym     = "ppm_generate_photon_paths_001";
    const char* advance_photon_paths_sym      = "ppm_advance_photon_paths_001";
    const char* accum_flux_sym                = "ppm_accum_flux_001";

    void perfmodel_init(starpu_perfmodel* model) {
      memset(model, 0, sizeof(starpu_perfmodel));
    }

    void init(const Config* cpu_config,       const PtrFreeScene* cpu_scene, PtrFreeHashGrid* cpu_hash_grid,
              const CUDA::Config* gpu_config, const PtrFreeScene* gpu_scene, PtrFreeHashGrid* gpu_hash_grid) {
      generic_args.cpu_config    = cpu_config;
      generic_args.cpu_scene     = cpu_scene;
      generic_args.cpu_hash_grid = cpu_hash_grid;
      generic_args.gpu_config    = gpu_config;
      generic_args.gpu_scene     = gpu_scene;
      generic_args.gpu_hash_grid = gpu_hash_grid;


      starpu_perfmodel* pm;
      struct starpu_codelet* cl;

      // generate_eye_paths
      pm = &generate_eye_paths_pm;
      perfmodel_init(pm);
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
      cl->modes[0]        = STARPU_RW; // eye_paths
      cl->modes[1]        = STARPU_RW; // seeds
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
      cl->nbuffers        = 3;
      cl->modes[0]        = STARPU_RW; // hit points
      cl->modes[1]        = STARPU_RW; // eye_paths
      cl->modes[2]        = STARPU_RW; // seed_buffer
      cl->model           = pm;


      // bbox_compute
      pm = &bbox_compute_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = bbox_compute_sym;

      cl = &bbox_compute;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->max_parallelism = 1;
      cl->cpu_funcs[0]    = ppm::kernels::cpu::bbox_compute;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_R;     // hit_points_info
      cl->modes[1]        = STARPU_REDUX; // bbox buffer
      cl->model           = pm;


      // bbox_zero_initialize
      pm = &bbox_zero_initialize_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = bbox_zero_initialize_sym;

      cl = &bbox_zero_initialize;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->max_parallelism = 1;
      cl->cpu_funcs[0]    = ppm::kernels::cpu::bbox_zero_initialize;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 1;
      cl->modes[0]        = STARPU_RW; // the bbox
      cl->model           = pm;


      // bbox_reduce
      pm = &bbox_reduce_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = bbox_reduce_sym;

      cl = &bbox_reduce;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->max_parallelism = 1;
      cl->cpu_funcs[0]    = ppm::kernels::cpu::bbox_reduce;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_RW; // bbox_dst
      cl->modes[1]        = STARPU_R;  // bbox_src
      cl->model           = pm;


      // rehash
      pm = &rehash_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = rehash_sym;

      cl = &rehash;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->max_parallelism = 1;
      cl->cpu_funcs[0]    = ppm::kernels::cpu::rehash;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_R; // hit_points_info
      cl->modes[1]        = STARPU_W; // entry_count
      //cl->modes[2]        = STARPU_R; // bbox
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
      cl->nbuffers        = 3;
      cl->modes[0]        = STARPU_RW; // hit points
      cl->modes[1]        = STARPU_RW; // eye_paths
      cl->modes[2]        = STARPU_RW; // seed_buffer
      cl->model           = pm;


      // generate_photon_paths
      pm = &generate_photon_paths_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = generate_photon_paths_sym;

      cl = &generate_photon_paths;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::generate_photon_paths;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_RW; // live_photon_paths
      cl->modes[1]        = STARPU_RW; // seeds
      cl->model           = pm;


      // advance_photon_paths
      pm = &advance_photon_paths_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = advance_photon_paths_sym;

      cl = &advance_photon_paths;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::advance_photon_paths;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 4;
      cl->modes[0]        = STARPU_RW; // live_photon_paths
      cl->modes[1]        = STARPU_R;  // hit_points_info
      cl->modes[2]        = STARPU_RW; // hit_points
      cl->modes[3]        = STARPU_RW; // seeds
      cl->model           = pm;


      // accum_flux
      pm = &accum_flux_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = accum_flux_sym;

      cl = &accum_flux;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::accum_flux;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_R;  // hit_points_info
      cl->modes[1]        = STARPU_RW; // hit_points
      cl->model           = pm;
    }
  }

} }
