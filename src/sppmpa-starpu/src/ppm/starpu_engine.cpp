#include "ppm/starpu_engine.h"
#include "ppm/kernels/codelets.h"
using namespace ppm::kernels;

namespace ppm {

//
// constructors
//

StarpuEngine :: StarpuEngine(const Config& _config)
: Engine(_config),

  sample_buffer(new SampleBuffer(config.width * config.height * config.spp * config.spp)),
  frame_buffer(new SampleFrameBuffer(config.width, config.height)),

  seeds(max(config.total_hit_points, config.photons_per_iter)),
  eye_paths(config.total_hit_points),
  hit_points_info(config.total_hit_points),
  hit_points(config.total_hit_points),
  live_photon_paths(config.photons_per_iter) {

  // load starpuk
  starpu_conf_init(&this->spu_conf);
  spu_conf.sched_policy_name = config.sched_policy.c_str();

  starpu_init(&this->spu_conf);
  kernels::codelets::init(&config, scene, &hash_grid, NULL, NULL, NULL); // TODO GPU versions here

  init_starpu_handles();
}

StarpuEngine :: ~StarpuEngine() {
  starpu_task_wait_for_all();
  starpu_shutdown();
}

//
// public methods
//
void StarpuEngine :: render() {
  start_time = WallClockTime();

  this->init_seed_buffer();

  // main loop
  while((!display || display->is_on()) && iteration <= config.max_iters) {
    this->generate_eye_paths();
    this->advance_eye_paths();
    this->bbox_compute();
    this->rehash();
    this->generate_photon_paths();
    this->advance_photon_paths();
    this->accumulate_flux();
    this->update_sample_buffer();
    this->splat_to_film();

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
  starpu_task_wait_for_all();
}

void StarpuEngine :: output() {
  film->SaveImpl(config.output_file);
}

//
// private methods
//

void StarpuEngine :: init_starpu_handles() {
  starpu_vector_data_register(&seeds_h,             -1, (uintptr_t)NULL,             seeds.size(),             sizeof(Seed));
  starpu_vector_data_register(&eye_paths_h,         -1, (uintptr_t)NULL,         eye_paths.size(),         sizeof(EyePath));
  starpu_vector_data_register(&hit_points_info_h,   -1, (uintptr_t)NULL,   hit_points_info.size(),   sizeof(HitPointPosition));
  starpu_vector_data_register(&hit_points_h,        0 , (uintptr_t)&hit_points[0],        hit_points.size(),        sizeof(HitPointPosition));
  starpu_vector_data_register(&live_photon_paths_h, -1, (uintptr_t)NULL, live_photon_paths.size(), sizeof(PhotonPath));

  starpu_variable_data_register(&bbox_h, 0, (uintptr_t)&bbox, sizeof(bbox));
  starpu_variable_data_register(&hash_grid_entry_count_h,  0, (uintptr_t)&hash_grid.entry_count,  sizeof(hash_grid.entry_count));
  starpu_variable_data_register(&current_photon_radius2_h, 0, (uintptr_t)&current_photon_radius2, sizeof(current_photon_radius2));
  starpu_variable_data_register(&sample_buffer_h,          0, (uintptr_t)&sample_buffer,          sizeof(sample_buffer));
  starpu_variable_data_register(&frame_buffer_h,           0, (uintptr_t)&frame_buffer,           sizeof(frame_buffer));
  starpu_variable_data_register(&film_h,                   0, (uintptr_t)&film,                   sizeof(film));
}

void StarpuEngine :: init_seed_buffer() {
  starpu_insert_task(&codelets::init_seeds,
    STARPU_W, seeds_h,
    STARPU_VALUE, &iteration, sizeof(iteration),
    0);
}

void StarpuEngine :: generate_eye_paths() {
  starpu_insert_task(&codelets::generate_eye_paths,
    STARPU_W, eye_paths_h,
    STARPU_RW, seeds_h,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    0);

}
void StarpuEngine :: advance_eye_paths() {
  starpu_insert_task(&codelets::advance_eye_paths,
    STARPU_W,  hit_points_info_h,
    STARPU_R,  eye_paths_h,
    STARPU_RW, seeds_h,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    0);
}

void StarpuEngine :: bbox_compute() {
  unsigned total_spp = config.width * config.spp + config.height * config.spp;

  starpu_insert_task(&codelets::bbox_compute,
    STARPU_R, hit_points_info_h,
    STARPU_W, bbox_h,
    STARPU_W, current_photon_radius2_h,
    STARPU_VALUE, &iteration,    sizeof(iteration),
    STARPU_VALUE, &total_spp,    sizeof(total_spp),
    STARPU_VALUE, &config.alpha, sizeof(config.alpha),
    0);
}

void StarpuEngine :: rehash() {
  starpu_insert_task(&codelets::rehash,
    STARPU_R,     hit_points_info_h,
    STARPU_R,     bbox_h,
    STARPU_R,     current_photon_radius2_h,
    STARPU_W,     hash_grid_entry_count_h,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    0);
}

void StarpuEngine :: generate_photon_paths() {
  starpu_insert_task(&codelets::generate_photon_paths,
    STARPU_W,  live_photon_paths_h,
    STARPU_RW, seeds_h,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    0);
}

void StarpuEngine :: advance_photon_paths() {
  starpu_insert_task(&codelets::advance_photon_paths,
    STARPU_R,  live_photon_paths_h,
    STARPU_R,  hit_points_info_h,
    STARPU_W,  hit_points_h,
    STARPU_RW, seeds_h,
    STARPU_R,  bbox_h,
    STARPU_R,  current_photon_radius2_h,
    STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
    0);
}

void StarpuEngine :: accumulate_flux() {
  starpu_insert_task(&codelets::accum_flux,
    STARPU_R,  hit_points_info_h,
    STARPU_RW, hit_points_h,
    STARPU_R,  current_photon_radius2_h,
    STARPU_VALUE, &codelets::generic_args,  sizeof(codelets::generic_args),
    STARPU_VALUE, &config.photons_per_iter, sizeof(config.photons_per_iter),
    0);
}


void StarpuEngine :: update_sample_buffer() {
  starpu_insert_task(&codelets::update_sample_buffer,
    STARPU_R,  hit_points_h,
    STARPU_RW, sample_buffer_h,
    STARPU_VALUE, &config.width, sizeof(config.width),
    0);
}

void StarpuEngine :: splat_to_film() {
  starpu_insert_task(&codelets::splat_to_film,
    STARPU_R,  sample_buffer_h,
    STARPU_RW, film_h,
    STARPU_VALUE, &config.width, sizeof(config.width),
    STARPU_VALUE, &config.height, sizeof(config.height),
    0);
}

}
