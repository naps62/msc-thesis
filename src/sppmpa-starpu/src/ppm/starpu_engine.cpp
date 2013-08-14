#include "ppm/starpu_engine.h"
#include "ppm/kernels/codelets.h"
using namespace ppm::kernels;

namespace ppm {

//
// constructors
//

void callback(void *arg) {
  printf("callback: %ld. left: %d, ready: %d\n", (long int)arg, starpu_task_nsubmitted(), starpu_task_nready());
  fflush(stdout);
}

StarpuEngine :: StarpuEngine(const Config& _config)
: Engine(_config) {

  // load starpuk
  starpu_conf_init(&this->spu_conf);
  spu_conf.sched_policy_name = config.sched_policy.c_str();

  PtrFreeScene* device_scene = scene->to_device(0);
  starpu_init(&this->spu_conf);

  kernels::codelets::init(&config, scene, device_scene); // TODO GPU versions here

}

StarpuEngine :: ~StarpuEngine() {
  starpu_task_wait_for_all();
  starpu_shutdown();
}

//
// public methods
//
void StarpuEngine :: before() {
  starpu_variable_data_register(&sample_buffer_h,  0, (uintptr_t)&sample_buffer,         sizeof(sample_buffer));
  starpu_variable_data_register(&film_h,           0, (uintptr_t)&film,                  sizeof(film));
  total_spp = config.width * config.spp + config.height * config.spp;

  seeds_h                   = new starpu_data_handle_t[config.max_iters_at_once];
  eye_paths_h               = new starpu_data_handle_t[config.max_iters_at_once];
  hit_points_info_h         = new starpu_data_handle_t[config.max_iters_at_once];
  hit_points_h              = new starpu_data_handle_t[config.max_iters_at_once];
  live_photon_paths_h       = new starpu_data_handle_t[config.max_iters_at_once];
  bbox_h                    = new starpu_data_handle_t[config.max_iters_at_once];
  current_photon_radius2_h  = new starpu_data_handle_t[config.max_iters_at_once];
  hash_grid_h               = new starpu_data_handle_t[config.max_iters_at_once];
  hash_grid_lengths_h       = new starpu_data_handle_t[config.max_iters_at_once];
  hash_grid_indexes_h       = new starpu_data_handle_t[config.max_iters_at_once];
  hash_grid_inv_cell_size_h = new starpu_data_handle_t[config.max_iters_at_once];

  for(unsigned i = 0; i < config.max_iters_at_once; ++i) {
    vector_handle(&seeds_h[i],             config.seed_size,            sizeof(Seed));
    vector_handle(&eye_paths_h[i],         config.total_hit_points,     sizeof(EyePath));
    vector_handle(&hit_points_info_h[i],   config.total_hit_points,     sizeof(HitPointPosition));
    vector_handle(&hash_grid_h[i],         8 * config.total_hit_points, sizeof(unsigned));
    vector_handle(&hash_grid_lengths_h[i], config.total_hit_points,     sizeof(unsigned));
    vector_handle(&hash_grid_indexes_h[i], config.total_hit_points,     sizeof(unsigned));
    vector_handle(&live_photon_paths_h[i], config.photons_per_iter,     sizeof(PhotonPath));
    vector_handle(&hit_points_h[i],        config.total_hit_points,     sizeof(HitPointRadiance));
    variable_handle(&bbox_h[i],                    sizeof(BBox));
    variable_handle(&current_photon_radius2_h[i],  sizeof(float));
    variable_handle(&hash_grid_inv_cell_size_h[i], sizeof(float));
  }

  handle_index = 0;
}

void StarpuEngine:: after() {
  free_handle(sample_buffer_h);
  free_handle(film_h);

  for(unsigned i = 0; i < config.max_iters_at_once; ++i) {
    free_handle(seeds_h[i]);
    free_handle(eye_paths_h[i]);
    free_handle(bbox_h[i]);
    free_handle(hash_grid_h[i]);
    free_handle(hash_grid_lengths_h[i]);
    free_handle(hash_grid_indexes_h[i]);
    free_handle(hash_grid_inv_cell_size_h[i]);
    free_handle(live_photon_paths_h[i]);
    free_handle(hit_points_info_h[i]);
    free_handle(current_photon_radius2_h[i]);
    free_handle(hit_points_h[i]);
  }

  delete[] seeds_h;
  delete[] eye_paths_h;
  delete[] hit_points_info_h;
  delete[] hit_points_h;
  delete[] live_photon_paths_h;
  delete[] bbox_h;
  delete[] current_photon_radius2_h;
  delete[] hash_grid_h;
  delete[] hash_grid_lengths_h;
  delete[] hash_grid_indexes_h;
  delete[] hash_grid_inv_cell_size_h;
}

void StarpuEngine :: init_seed_buffer() {
  for(unsigned i = 0; i < config.max_iters_at_once; ++i) {
    starpu_insert_task(&codelets::init_seeds,
      STARPU_W, seeds_h[i],
      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
      STARPU_VALUE, &iteration, sizeof(iteration),
      STARPU_CALLBACK_WITH_ARG, callback, (void*)1, 0);
  }
}

void StarpuEngine :: generate_eye_paths() {
  starpu_insert_task( &codelets::generate_eye_paths,
                      STARPU_W,  eye_paths_h[handle_index],
                      STARPU_RW, seeds_h[handle_index],
                      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                      STARPU_CALLBACK_WITH_ARG, callback, (void*)2, 0);

}
void StarpuEngine :: advance_eye_paths() {
  starpu_insert_task( &codelets::advance_eye_paths,
                      STARPU_W,  hit_points_info_h[handle_index],
                      STARPU_R,  eye_paths_h[handle_index],
                      STARPU_RW, seeds_h[handle_index],
                      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                      STARPU_CALLBACK_WITH_ARG, callback, (void*)3, 0);
}

void StarpuEngine :: bbox_compute() {
    starpu_insert_task( &codelets::bbox_compute,
                        STARPU_R, hit_points_info_h[handle_index],
                        STARPU_W, bbox_h[handle_index],
                        STARPU_W, current_photon_radius2_h[handle_index],
                        STARPU_VALUE, &iteration,    sizeof(iteration),
                        STARPU_VALUE, &total_spp,    sizeof(total_spp),
                        STARPU_VALUE, &config.alpha, sizeof(config.alpha),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)4, 0);
}

void StarpuEngine :: rehash() {
    starpu_insert_task( &codelets::rehash,
                        STARPU_R,     hit_points_info_h[handle_index],
                        STARPU_R,     bbox_h[handle_index],
                        STARPU_R,     current_photon_radius2_h[handle_index],
                        STARPU_W,     hash_grid_h[handle_index],
                        STARPU_W,     hash_grid_lengths_h[handle_index],
                        STARPU_W,     hash_grid_indexes_h[handle_index],
                        STARPU_W,     hash_grid_inv_cell_size_h[handle_index],
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)5, 0);
}

void StarpuEngine :: generate_photon_paths() {
    starpu_insert_task( &codelets::generate_photon_paths,
                        STARPU_W,  live_photon_paths_h[handle_index],
                        STARPU_RW, seeds_h[handle_index],
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)6, 0);
}

void StarpuEngine :: advance_photon_paths() {
    starpu_insert_task( &codelets::advance_photon_paths,
                        STARPU_R,  live_photon_paths_h[handle_index],
                        STARPU_R,  hit_points_info_h[handle_index],
                        STARPU_W,  hit_points_h[handle_index],
                        STARPU_RW, seeds_h[handle_index],
                        STARPU_R,  bbox_h[handle_index],
                        STARPU_R,  current_photon_radius2_h[handle_index],
                        STARPU_R,  hash_grid_h[handle_index],
                        STARPU_R,  hash_grid_lengths_h[handle_index],
                        STARPU_R,  hash_grid_indexes_h[handle_index],
                        STARPU_R,  hash_grid_inv_cell_size_h[handle_index],
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                        STARPU_VALUE, &config.total_hit_points, sizeof(config.total_hit_points),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)7, 0);
}

void StarpuEngine :: accumulate_flux() {
    starpu_insert_task( &codelets::accum_flux,
                        STARPU_R,  hit_points_info_h[handle_index],
                        STARPU_RW, hit_points_h[handle_index],
                        STARPU_R,  current_photon_radius2_h[handle_index],
                        STARPU_VALUE, &codelets::generic_args,  sizeof(codelets::generic_args),
                        STARPU_VALUE, &config.alpha,  sizeof(config.alpha),
                        STARPU_VALUE, &config.photons_per_iter, sizeof(config.photons_per_iter),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)8, 0);
}


void StarpuEngine :: update_sample_buffer() {
    starpu_insert_task( &codelets::update_sample_buffer,
                        STARPU_R,  hit_points_h[handle_index],
                        STARPU_RW, sample_buffer_h,
                        STARPU_VALUE, &config.width, sizeof(config.width),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)9, 0);

  handle_index++;
  if (handle_index == config.max_iters_at_once)
    handle_index = 0;
}

void StarpuEngine :: splat_to_film() {
    starpu_insert_task( &codelets::splat_to_film,
                        STARPU_R,  sample_buffer_h,
                        STARPU_RW, film_h,
                        STARPU_VALUE, &config.width, sizeof(config.width),
                        STARPU_VALUE, &config.height, sizeof(config.height),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)10, 0);
}

void StarpuEngine :: wait_for_all() {
  starpu_task_wait_for_all();
}

//
// private methods
//


void StarpuEngine :: vector_handle(starpu_data_handle_t* handle, unsigned total, size_t size) {
  starpu_vector_data_register(handle, -1, (uintptr_t)NULL, total, size);
}

void StarpuEngine :: variable_handle(starpu_data_handle_t* handle, size_t size) {
  starpu_variable_data_register(handle, -1, (uintptr_t)NULL, size);
}

void StarpuEngine :: free_handle(starpu_data_handle_t handle) {
  starpu_data_unregister_submit(handle);
}

}
