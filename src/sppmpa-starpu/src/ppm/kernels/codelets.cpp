#include "ppm/kernels/codelets.h"
#include "ppm/kernels/headers.h"

#include <limits>


namespace ppm { namespace kernels {

  namespace codelets {
    starpu_args    generic_args;

    struct starpu_codelet init_seeds;
    struct starpu_codelet generate_eye_paths;
    struct starpu_codelet advance_eye_paths;
    struct starpu_codelet bbox_compute;
    struct starpu_codelet rehash;
    struct starpu_codelet generate_photon_paths;
    struct starpu_codelet advance_photon_paths;
    struct starpu_codelet accum_flux;
    struct starpu_codelet update_sample_buffer;
    struct starpu_codelet splat_to_film;

    starpu_perfmodel init_seeds_pm;
    starpu_perfmodel generate_eye_paths_pm;
    starpu_perfmodel advance_eye_paths_pm;
    starpu_perfmodel bbox_compute_pm;
    starpu_perfmodel rehash_pm;
    starpu_perfmodel generate_photon_paths_pm;
    starpu_perfmodel advance_photon_paths_pm;
    starpu_perfmodel accum_flux_pm;
    starpu_perfmodel update_sample_buffer_pm;
    starpu_perfmodel splat_to_film_pm;

    const char* init_seeds_sym            = "ppm_init_seeds_001";
    const char* generate_eye_paths_sym    = "ppm_generate_eye_paths_001";
    const char* advance_eye_paths_sym     = "ppm_advance_eye_paths_001";
    const char* bbox_compute_sym          = "ppm_bbox_compute_001";
    const char* rehash_sym                = "ppm_rehash_001";
    const char* generate_photon_paths_sym = "ppm_generate_photon_paths_001";
    const char* advance_photon_paths_sym  = "ppm_advance_photon_paths_001";
    const char* accum_flux_sym            = "ppm_accum_flux_001";
    const char* update_sample_buffer_sym  = "ppm_update_sample_frame_buffer_001";
    const char* splat_to_film_sym         = "ppm_splat_to_film_001";

    void perfmodel_init(starpu_perfmodel* model) {
      memset(model, 0, sizeof(starpu_perfmodel));
    }

    void init(const Config* config, const PtrFreeScene* cpu_scene, const PtrFreeScene* gpu_scene) {
      generic_args.config    = config;
      generic_args.cpu_scene = cpu_scene;
      generic_args.gpu_scene = gpu_scene;


      starpu_perfmodel* pm;
      struct starpu_codelet* cl;


      // init_seeds
      pm = &init_seeds_pm;
      perfmodel_init(pm);
      pm->type   = STARPU_HISTORY_BASED;
      pm->symbol = init_seeds_sym;

      cl   = &init_seeds;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::init_seeds;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 1;
      cl->modes[0]        = STARPU_W; // seeds
      cl->model           = pm;


      // generate_eye_paths
      pm = &generate_eye_paths_pm;
      perfmodel_init(pm);
      pm->type   = STARPU_HISTORY_BASED;
      pm->symbol = generate_eye_paths_sym;

      cl   = &generate_eye_paths;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU | STARPU_CUDA;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::generate_eye_paths;
      cl->cpu_funcs[1]    = NULL;
      cl->cuda_funcs[0]   = ppm::kernels::cuda::generate_eye_paths;
      cl->cuda_funcs[1]   = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_W;  // eye_paths
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
      cl->modes[0]        = STARPU_W;  // hit points
      cl->modes[1]        = STARPU_R;  // eye_paths
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
      cl->nbuffers        = 3;
      cl->modes[0]        = STARPU_R; // hit_points_info
      cl->modes[1]        = STARPU_W; // bbox buffer
      cl->modes[2]        = STARPU_W; // photon_radius2
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
      cl->nbuffers        = 7;
      cl->modes[0]        = STARPU_R; // hit_points_info
      cl->modes[1]        = STARPU_R; // bbox
      cl->modes[2]        = STARPU_R; // current_photon_radius2
      cl->modes[3]        = STARPU_W; // hash_grid_ptr
      cl->modes[4]        = STARPU_W; // hash_grid_lengths
      cl->modes[5]        = STARPU_W; // hash_grid_indexes
      cl->modes[6]        = STARPU_W; // hash_grid_inv_cell_size
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
      cl->modes[0]        = STARPU_W;  // live_photon_paths
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
      cl->nbuffers        = 10;
      cl->modes[0]        = STARPU_R;  // live_photon_paths
      cl->modes[1]        = STARPU_R;  // hit_points_info
      cl->modes[2]        = STARPU_W;  // hit_points
      cl->modes[3]        = STARPU_RW; // seeds
      cl->modes[4]        = STARPU_R;  // bbox
      cl->modes[5]        = STARPU_R;  // current_photon_radius2
      cl->modes[6]        = STARPU_R;  // hash_grid_ptr
      cl->modes[7]        = STARPU_R;  // hash_grid_lengths
      cl->modes[8]        = STARPU_R;  // hash_grid_indexes
      cl->modes[9]        = STARPU_R;  // hash_grid_inv_cell_size
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
      cl->nbuffers        = 3;
      cl->modes[0]        = STARPU_R;  // hit_points_info
      cl->modes[1]        = STARPU_RW; // hit_points
      cl->modes[2]        = STARPU_R;  // photon_radius2
      cl->model           = pm;



      // update_sample_buffer
      pm = &update_sample_buffer_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = update_sample_buffer_sym;

      cl = &update_sample_buffer;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->type            = STARPU_FORKJOIN;
      cl->max_parallelism = std::numeric_limits<int>::max();
      cl->cpu_funcs[0]    = ppm::kernels::cpu::update_sample_buffer;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_R;  // hit_points
      cl->modes[1]        = STARPU_RW; // sample_buffer
      cl->model           = pm;


      // splat_to_film
      pm = &splat_to_film_pm;
      perfmodel_init(pm);
      pm->type = STARPU_HISTORY_BASED;
      pm->symbol = splat_to_film_sym;

      cl = &splat_to_film;
      starpu_codelet_init(cl);
      cl->where           = STARPU_CPU;
      cl->max_parallelism = 1;
      cl->cpu_funcs[0]    = ppm::kernels::cpu::splat_to_film;
      cl->cpu_funcs[1]    = NULL;
      cl->nbuffers        = 2;
      cl->modes[0]        = STARPU_R;  // sample_buffer
      cl->modes[1]        = STARPU_RW; // film
      cl->model           = pm;
    }
  }

} }
