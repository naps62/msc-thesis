#include "ppm/cpu_engine.h"
#include "ppm/kernels/codelets.h"
using namespace ppm::kernels;

namespace ppm {

//
// constructors
//

CPUEngine :: CPUEngine(const Config& _config)
: Engine(_config),

  frame_buffer(new SampleFrameBuffer(config.width, config.height)),

  seeds(max(config.total_hit_points, config.photons_per_iter)),
  eye_paths(config.total_hit_points),
  hit_points_info(config.total_hit_points),
  hit_points(config.total_hit_points),
  live_photon_paths(config.photons_per_iter) {
}

CPUEngine :: ~CPUEngine() {
  delete sample_buffer;
  delete frame_buffer;
}

//
// public methods
//
void CPUEngine :: render() {
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

    if (display) {
      set_captions();
      display->request_update(config.min_frame_time);
    }
  }
}

//
// private methods
//

void CPUEngine :: init_seed_buffer() {
  // starpu_insert_task(&codelets::init_seeds,
  //   STARPU_W, seeds_h,
  //   STARPU_VALUE, &iteration, sizeof(iteration),
  //   0);
}

void CPUEngine :: generate_eye_paths() {
  // starpu_insert_task(&codelets::generate_eye_paths,
  //   STARPU_W, eye_paths_h,
  //   STARPU_RW, seeds_h,
  //   STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
  //   0);

}
void CPUEngine :: advance_eye_paths() {
  // starpu_insert_task(&codelets::advance_eye_paths,
  //   STARPU_W,  hit_points_info_h,
  //   STARPU_R,  eye_paths_h,
  //   STARPU_RW, seeds_h,
  //   STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
  //   0);
}

void CPUEngine :: bbox_compute() {
  // unsigned total_spp = config.width * config.spp + config.height * config.spp;

  // starpu_insert_task(&codelets::bbox_compute,
  //   STARPU_R, hit_points_info_h,
  //   STARPU_W, bbox_h,
  //   STARPU_W, current_photon_radius2_h,
  //   STARPU_VALUE, &iteration,    sizeof(iteration),
  //   STARPU_VALUE, &total_spp,    sizeof(total_spp),
  //   STARPU_VALUE, &config.alpha, sizeof(config.alpha),
  //   0);
}

void CPUEngine :: rehash() {
  // starpu_insert_task(&codelets::rehash,
  //   STARPU_R,     hit_points_info_h,
  //   STARPU_R,     bbox_h,
  //   STARPU_R,     current_photon_radius2_h,
  //   STARPU_W,     hash_grid_entry_count_h,
  //   STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
  //   0);
}

void CPUEngine :: generate_photon_paths() {
  // starpu_insert_task(&codelets::generate_photon_paths,
  //   STARPU_W,  live_photon_paths_h,
  //   STARPU_RW, seeds_h,
  //   STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
  //   0);
}

void CPUEngine :: advance_photon_paths() {
  // starpu_insert_task(&codelets::advance_photon_paths,
  //   STARPU_R,  live_photon_paths_h,
  //   STARPU_R,  hit_points_info_h,
  //   STARPU_W,  hit_points_h,
  //   STARPU_RW, seeds_h,
  //   STARPU_R,  bbox_h,
  //   STARPU_R,  current_photon_radius2_h,
  //   STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
  //   0);
}

void CPUEngine :: accumulate_flux() {
  // starpu_insert_task(&codelets::accum_flux,
  //   STARPU_R,  hit_points_info_h,
  //   STARPU_RW, hit_points_h,
  //   STARPU_R,  current_photon_radius2_h,
  //   STARPU_VALUE, &codelets::generic_args,  sizeof(codelets::generic_args),
  //   STARPU_VALUE, &config.photons_per_iter, sizeof(config.photons_per_iter),
  //   0);
}


void CPUEngine :: update_sample_buffer() {
  // starpu_insert_task(&codelets::update_sample_buffer,
  //   STARPU_R,  hit_points_h,
  //   STARPU_RW, sample_buffer_h,
  //   STARPU_VALUE, &config.width, sizeof(config.width),
  //   0);
}

void CPUEngine :: splat_to_film() {
  // starpu_insert_task(&codelets::splat_to_film,
  //   STARPU_R,  sample_buffer_h,
  //   STARPU_RW, film_h,
  //   STARPU_VALUE, &config.width, sizeof(config.width),
  //   STARPU_VALUE, &config.height, sizeof(config.height),
  //   0);
}

}
