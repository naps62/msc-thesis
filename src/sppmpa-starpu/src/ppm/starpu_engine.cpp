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

  vector_handle(&seeds_h, config.seed_size, sizeof(Seed));
  vector_handle(&eye_paths_h, config.total_hit_points, sizeof(EyePath));
  vector_handle(&hit_points_info_h, config.total_hit_points, sizeof(HitPointPosition));
  variable_handle(&bbox_h,                   sizeof(BBox));
  variable_handle(&current_photon_radius2_h, sizeof(float));
  vector_handle(&hash_grid_h,         8 * config.total_hit_points, sizeof(unsigned));
  vector_handle(&hash_grid_lengths_h, config.total_hit_points,     sizeof(unsigned));
  vector_handle(&hash_grid_indexes_h, config.total_hit_points,     sizeof(unsigned));
  variable_handle(&hash_grid_inv_cell_size_h, sizeof(float));
  vector_handle(&live_photon_paths_h, config.photons_per_iter, sizeof(PhotonPath));
  vector_handle(&hit_points_h, config.total_hit_points, sizeof(HitPointRadiance));
}

void StarpuEngine:: after() {
  free_handle(sample_buffer_h);
  free_handle(film_h);
  free_handle(seeds_h);
  free_handle(eye_paths_h);
  free_handle(bbox_h);
  free_handle(hash_grid_h);
  free_handle(hash_grid_lengths_h);
  free_handle(hash_grid_indexes_h);
  free_handle(hash_grid_inv_cell_size_h);
  free_handle(live_photon_paths_h);
  free_handle(hit_points_info_h);
  free_handle(current_photon_radius2_h);
  free_handle(hit_points_h);
}

void StarpuEngine :: init_seed_buffer() {
  starpu_insert_task(&codelets::init_seeds,
    STARPU_W, seeds_h,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    STARPU_VALUE, &iteration, sizeof(iteration),
    STARPU_CALLBACK_WITH_ARG, callback, (void*)1, 0);
}

void StarpuEngine :: generate_eye_paths() {
  starpu_insert_task( &codelets::generate_eye_paths,
                      STARPU_W,  eye_paths_h,
                      STARPU_RW, seeds_h,
                      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                      STARPU_CALLBACK_WITH_ARG, callback, (void*)2, 0);

}
void StarpuEngine :: advance_eye_paths() {
  starpu_insert_task( &codelets::advance_eye_paths,
                      STARPU_W,  hit_points_info_h,
                      STARPU_R,  eye_paths_h,
                      STARPU_RW, seeds_h,
                      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                      STARPU_CALLBACK_WITH_ARG, callback, (void*)3, 0);
}

void StarpuEngine :: bbox_compute() {
    starpu_insert_task( &codelets::bbox_compute,
                        STARPU_R, hit_points_info_h,
                        STARPU_W, bbox_h,
                        STARPU_W, current_photon_radius2_h,
                        STARPU_VALUE, &iteration,    sizeof(iteration),
                        STARPU_VALUE, &total_spp,    sizeof(total_spp),
                        STARPU_VALUE, &config.alpha, sizeof(config.alpha),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)4, 0);
}

void StarpuEngine :: rehash() {
    starpu_insert_task( &codelets::rehash,
                        STARPU_R,     hit_points_info_h,
                        STARPU_R,     bbox_h,
                        STARPU_R,     current_photon_radius2_h,
                        STARPU_W,     hash_grid_h,
                        STARPU_W,     hash_grid_lengths_h,
                        STARPU_W,     hash_grid_indexes_h,
                        STARPU_W,     hash_grid_inv_cell_size_h,
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)5, 0);
}

void StarpuEngine :: generate_photon_paths() {
    starpu_insert_task( &codelets::generate_photon_paths,
                        STARPU_W,  live_photon_paths_h,
                        STARPU_RW, seeds_h,
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)6, 0);
}

void StarpuEngine :: advance_photon_paths() {
    starpu_insert_task( &codelets::advance_photon_paths,
                        STARPU_R,  live_photon_paths_h,
                        STARPU_R,  hit_points_info_h,
                        STARPU_W,  hit_points_h,
                        STARPU_RW, seeds_h,
                        STARPU_R,  bbox_h,
                        STARPU_R,  current_photon_radius2_h,
                        STARPU_R,  hash_grid_h,
                        STARPU_R,  hash_grid_lengths_h,
                        STARPU_R,  hash_grid_indexes_h,
                        STARPU_R,  hash_grid_inv_cell_size_h,
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
                        STARPU_VALUE, &config.total_hit_points, sizeof(config.total_hit_points),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)7, 0);
}

void StarpuEngine :: accumulate_flux() {
    starpu_insert_task( &codelets::accum_flux,
                        STARPU_R,  hit_points_info_h,
                        STARPU_RW, hit_points_h,
                        STARPU_R,  current_photon_radius2_h,
                        STARPU_VALUE, &codelets::generic_args,  sizeof(codelets::generic_args),
                        STARPU_VALUE, &config.photons_per_iter, sizeof(config.photons_per_iter),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)8, 0);
}


void StarpuEngine :: update_sample_buffer() {
    starpu_insert_task( &codelets::update_sample_buffer,
                        STARPU_R,  hit_points_h,
                        STARPU_RW, sample_buffer_h,
                        STARPU_VALUE, &config.width, sizeof(config.width),
                        STARPU_CALLBACK_WITH_ARG, callback, (void*)9, 0);
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
