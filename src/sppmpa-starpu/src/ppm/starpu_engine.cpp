#include "ppm/starpu_engine.h"
#include "ppm/kernels/codelets.h"
using namespace ppm::kernels;

namespace ppm {

//
// constructors
//

StarpuEngine :: StarpuEngine(const Config& _config)
: Engine(_config) {

  // load starpuk
  starpu_conf_init(&this->spu_conf);
  spu_conf.sched_policy_name = config.sched_policy.c_str();

  starpu_init(&this->spu_conf);
  kernels::codelets::init(&config, scene, NULL, NULL, NULL, NULL); // TODO GPU versions here

  init_starpu_handles();
}

StarpuEngine :: ~StarpuEngine() {
  starpu_task_wait_for_all();
  starpu_shutdown();
  delete sample_buffer;
}

//
// public methods
//
void StarpuEngine :: render() {
  start_time = WallClockTime();
  const unsigned total_spp = config.width * config.spp + config.height * config.spp;

  starpu_variable_data_register(&sample_buffer_h,  0, (uintptr_t)&sample_buffer,         sizeof(sample_buffer));
  starpu_variable_data_register(&film_h,           0, (uintptr_t)&film,                  sizeof(film));

  vector_handle(&seeds_h, config.seed_size, sizeof(Seed));

  // 1. INIT SEEDS
  starpu_insert_task(&codelets::init_seeds,
    STARPU_W, seeds_h,
    STARPU_VALUE, &iteration, sizeof(iteration),
    0);

  // main loop
  while((!display || display->is_on()) && iteration <= config.max_iters) {


    // 2. GENERATE EYE PATHS
    vector_handle(&eye_paths_h, config.total_hit_points, sizeof(EyePath));
    starpu_insert_task( &codelets::generate_eye_paths,
                        STARPU_W,  eye_paths_h,
                        STARPU_RW, seeds_h,
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args), 0);


    // 3. ADVANCE EYE PATHS
    vector_handle(&hit_points_info_h, config.total_hit_points, sizeof(HitPointPosition));
    starpu_insert_task( &codelets::advance_eye_paths,
                        STARPU_W,  hit_points_info_h,
                        STARPU_R,  eye_paths_h,
                        STARPU_RW, seeds_h,
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args), 0);
    free_handle(eye_paths_h);


    // 4. BBOX COMPUTE
    variable_handle(&bbox_h,                   sizeof(BBox));
    variable_handle(&current_photon_radius2_h, sizeof(float));
    starpu_insert_task( &codelets::bbox_compute,
                        STARPU_R, hit_points_info_h,
                        STARPU_W, bbox_h,
                        STARPU_W, current_photon_radius2_h,
                        STARPU_VALUE, &iteration,    sizeof(iteration),
                        STARPU_VALUE, &total_spp,    sizeof(total_spp),
                        STARPU_VALUE, &config.alpha, sizeof(config.alpha), 0);
    vector_handle(&hash_grid_h,         8 * config.total_hit_points, sizeof(unsigned));
    vector_handle(&hash_grid_lengths_h, config.total_hit_points,     sizeof(unsigned));
    vector_handle(&hash_grid_indexes_h, config.total_hit_points,     sizeof(unsigned));
    variable_handle(&hash_grid_inv_cell_size_h, sizeof(float));


    // 5. REHASH
    starpu_insert_task( &codelets::rehash,
                        STARPU_R,     hit_points_info_h,
                        STARPU_R,     bbox_h,
                        STARPU_R,     current_photon_radius2_h,
                        STARPU_W,     hash_grid_h,
                        STARPU_W,     hash_grid_lengths_h,
                        STARPU_W,     hash_grid_indexes_h,
                        STARPU_W,     hash_grid_inv_cell_size_h,
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args), 0);


    // 6. GENERATE PHOTON PATHS
    vector_handle(&live_photon_paths_h, config.photons_per_iter, sizeof(PhotonPath));
    starpu_insert_task( &codelets::generate_photon_paths,
                        STARPU_W,  live_photon_paths_h,
                        STARPU_RW, seeds_h,
                        STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args), 0);


    // 7. ADVANCE PHOTON PATHS
    vector_handle(&hit_points_h, config.total_hit_points, sizeof(HitPointRadiance));
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
                        STARPU_VALUE, &config.total_hit_points, sizeof(config.total_hit_points), 0);
    free_handle(bbox_h);
    free_handle(hash_grid_h);
    free_handle(hash_grid_lengths_h);
    free_handle(hash_grid_indexes_h);
    free_handle(hash_grid_inv_cell_size_h);
    free_handle(live_photon_paths_h);


    // 8. ACCUM FLUX
    starpu_insert_task( &codelets::accum_flux,
                        STARPU_R,  hit_points_info_h,
                        STARPU_RW, hit_points_h,
                        STARPU_R,  current_photon_radius2_h,
                        STARPU_VALUE, &codelets::generic_args,  sizeof(codelets::generic_args),
                        STARPU_VALUE, &config.photons_per_iter, sizeof(config.photons_per_iter), 0);
    free_handle(hit_points_info_h);
    free_handle(current_photon_radius2_h);


    // 9. UPDATE SAMPLE BUFFER
    starpu_insert_task( &codelets::update_sample_buffer,
                        STARPU_R,  hit_points_h,
                        STARPU_RW, sample_buffer_h,
                        STARPU_VALUE, &config.width, sizeof(config.width), 0);
    free_handle(hit_points_h);


    // 10. SPLAT TO FILM
    starpu_insert_task( &codelets::splat_to_film,
                        STARPU_R,  sample_buffer_h,
                        STARPU_RW, film_h,
                        STARPU_VALUE, &config.width, sizeof(config.width),
                        STARPU_VALUE, &config.height, sizeof(config.height), 0);



    total_photons_traced += config.photons_per_iter;
    iteration++;

    if ((config.max_iters_at_once > 0 && iteration % config.max_iters_at_once == 0)) {
      starpu_task_wait_for_all();
    } else if (config.max_iters_at_once == 0 && display) {
      starpu_task_wait_for_all();
    }

    if (display) {
      set_captions();
      display->request_update(config.min_frame_time);
    }
  }

  free_handle(sample_buffer_h);
  free_handle(film_h);
  free_handle(seeds_h);
  starpu_task_wait_for_all();
}

//
// private methods
//

void StarpuEngine :: init_starpu_handles() {

  // data partitions
  // filter_by_hit_points.filter_func = starpu_vector_filter_block;
  // filter_by_hit_points.nchildren   = config.total_hit_points / config.partition_size;
  // starpu_data_partition(eye_paths_h,       &filter_by_hit_points);
  // starpu_data_partition(hit_points_info_h, &filter_by_hit_points);
  //starpu_data_partition(hit_points,        &filter_by_hit_points);
}

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
