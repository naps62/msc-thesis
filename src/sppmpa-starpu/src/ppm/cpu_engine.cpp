#include "ppm/cpu_engine.h"
#include "ppm/kernels/codelets.h"
using namespace ppm::kernels;

namespace ppm {

//
// constructors
//

CPUEngine :: CPUEngine(const Config& _config)
: Engine(_config),

  seeds(max(config.total_hit_points, config.photons_per_iter)),
  eye_paths(config.total_hit_points),
  hit_points_info(config.total_hit_points),
  hit_points(config.total_hit_points),
  live_photon_paths(config.photons_per_iter),

  hash_grid(8 * config.total_hit_points),
  hash_grid_indexes(config.total_hit_points),
  hash_grid_lengths(config.total_hit_points) {
}

CPUEngine :: ~CPUEngine() {
}


//
// private methods
//

void CPUEngine :: init_seed_buffer() {
  const timeval start_time = my_WallClockTime();
  kernels::cpu::init_seeds_impl(&seeds[0], seeds.size(), iteration, config.max_threads);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, 0, config.max_threads, start_time, end_time, "(1) init_seeds");
}

void CPUEngine :: generate_eye_paths() {
  const timeval start_time = my_WallClockTime();
  kernels::cpu::generate_eye_paths_impl(&eye_paths[0],
                                        &seeds[0],
                                        config.width,
                                        config.height,
                                        scene,
                                        config.max_threads);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, config.max_threads, iteration, start_time, end_time, "(2) generate_eye_paths");
}

void CPUEngine :: advance_eye_paths() {
  const timeval start_time = my_WallClockTime();
  kernels::cpu::advance_eye_paths_impl(&hit_points_info[0],
                                       &eye_paths[0], eye_paths.size(),
                                       &seeds[0],
                                       scene,
                                       config.max_eye_path_depth,
                                       config.max_threads);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, config.max_threads, iteration, start_time, end_time, "(3) advance_eye_paths");

}

void CPUEngine :: bbox_compute() {
  const timeval start_time = my_WallClockTime();
  const unsigned total_spp = config.width * config.spp + config.height * config.spp;

  kernels::cpu::bbox_compute_impl(&hit_points_info[0], hit_points_info.size(),
                                  bbox,
                                  current_photon_radius2,
                                  iteration,
                                  total_spp,
                                  config.alpha);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, 1, iteration, start_time, end_time, "(4) bbox_compute");
}

void CPUEngine :: rehash() {
  const timeval start_time = my_WallClockTime();
  kernels::cpu::rehash_impl(&hit_points_info[0],
                            hit_points_info.size(),
                            &hash_grid[0],
                            &hash_grid_lengths[0],
                            &hash_grid_indexes[0],
                            &inv_cell_size,
                            bbox,
                            current_photon_radius2);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, 1, iteration, start_time, end_time, "(5) rehash");
}

void CPUEngine :: generate_photon_paths() {
  const timeval start_time = my_WallClockTime();
  kernels::cpu::generate_photon_paths_impl(&live_photon_paths[0], live_photon_paths.size(),
                                           &seeds[0],  // seed_buffer_count,
                                           scene,
                                           config.max_threads);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, config.max_threads, iteration, start_time, end_time, "(6) generate_photon_paths");


}

void CPUEngine :: advance_photon_paths() {
  const timeval start_time = my_WallClockTime();
  memset(&hit_points[0], 0, sizeof(HitPointRadiance) * config.total_hit_points);
  kernels::cpu::advance_photon_paths_impl(&live_photon_paths[0], live_photon_paths.size(),
                                          &seeds[0],  // seed_buffer_count,
                                          scene,
                                          &hit_points_info[0],
                                          &hit_points[0],
                                          bbox,
                                          config.max_photon_depth,
                                          current_photon_radius2,
                                          hit_points_info.size(),

                                          &hash_grid[0],
                                          &hash_grid_lengths[0],
                                          &hash_grid_indexes[0],
                                          inv_cell_size,
                                          config.max_threads);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, config.max_threads, iteration, start_time, end_time, "(7) advance_photon_paths");

}

void CPUEngine :: accumulate_flux() {
  const timeval start_time = my_WallClockTime();

  kernels::cpu::accum_flux_impl(&hit_points_info[0],
                                &hit_points[0],
                                hit_points.size(),
                                config.alpha,
                                config.photons_per_iter,
                                current_photon_radius2,
                                config.max_threads);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, config.max_threads, iteration, start_time, end_time, "(8) accum_flux");
}


void CPUEngine :: update_sample_buffer() {
  const timeval start_time = my_WallClockTime();
  kernels::cpu::update_sample_buffer_impl(&hit_points[0], hit_points.size(), config.width, sample_buffer, config.max_threads);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, config.max_threads, iteration, start_time, end_time, "(9) update_sample_buffer");
}

void CPUEngine :: splat_to_film() {
  const timeval start_time = my_WallClockTime();
  kernels::cpu::splat_to_film_impl(sample_buffer, film, config.width, config.height);
  const timeval end_time = my_WallClockTime();
  task_info("CPU", 0, 1, iteration, start_time, end_time, "(10) splat_to_film");
}

void CPUEngine :: before() {
  memset(&seeds[0],         0, sizeof(Seed)*seeds.size());
  memset(&eye_paths[0],         0, sizeof(EyePath)*eye_paths.size());
  memset(&live_photon_paths[0], 0, sizeof(PhotonPath)*live_photon_paths.size());
  memset(&hit_points_info[0],   0, sizeof(HitPointPosition)*hit_points.size());
  memset(&hit_points[0],        0, sizeof(HitPointRadiance)*hit_points.size());
  memset(&hash_grid[0], 0, sizeof(unsigned) * hash_grid.size());
  memset(&hash_grid_indexes[0], 0, sizeof(unsigned) * hash_grid_indexes.size());
  memset(&hash_grid_lengths[0], 0, sizeof(unsigned) * hash_grid_lengths.size());
  sample_buffer->Reset();
}

void CPUEngine :: after() {

}

void CPUEngine :: wait_for_all() {

}

}
