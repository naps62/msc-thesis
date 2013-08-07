#include "ppm/cuda_engine.h"
#include "ppm/kernels/codelets.h"
using namespace ppm::kernels;

namespace ppm {

//
// constructors
//

CUDAEngine :: CUDAEngine(const Config& _config)
: Engine(_config),
  cuda_scene(scene->to_device(0)) {

}

CUDAEngine :: ~CUDAEngine() {
}


//
// private methods
//

void CUDAEngine :: init_seed_buffer() {
  //kernels::cpu::init_seeds_impl(&seeds[0], seeds.size(), iteration);
}

void CUDAEngine :: generate_eye_paths() {
  /*kernels::cpu::generate_eye_paths_impl(&eye_paths[0],
                                        &seeds[0],
                                        config.width,
                                        config.height,
                                        scene);*/
}
void CUDAEngine :: advance_eye_paths() {
  /*kernels::cpu::advance_eye_paths_impl(&hit_points_info[0],
                                       &eye_paths[0], eye_paths.size(),
                                       &seeds[0],
                                       scene,
                                       config.max_eye_path_depth);*/
}

void CUDAEngine :: bbox_compute() {
  /*const unsigned total_spp = config.width * config.spp + config.height * config.spp;

  kernels::cpu::bbox_compute_impl(&hit_points_info[0], hit_points_info.size(),
                                  bbox,
                                  current_photon_radius2,
                                  iteration,
                                  total_spp,
                                  config.alpha);*/
}

void CUDAEngine :: rehash() {
  /*kernels::cpu::rehash_impl(&hit_points_info[0],
                            hit_points_info.size(),
                            &hash_grid[0],
                            &hash_grid_lengths[0],
                            &hash_grid_indexes[0],
                            &inv_cell_size,
                            bbox,
                            current_photon_radius2);*/

}

void CUDAEngine :: generate_photon_paths() {
  /*kernels::cpu::generate_photon_paths_impl(&live_photon_paths[0], live_photon_paths.size(),
                                           &seeds[0],  // seed_buffer_count,
                                           scene);*/
}

void CUDAEngine :: advance_photon_paths() {
  /*kernels::cpu::advance_photon_paths_impl(&live_photon_paths[0], live_photon_paths.size(),
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
                                          inv_cell_size);*/
}

void CUDAEngine :: accumulate_flux() {
  /*kernels::cpu::accum_flux_impl(&hit_points_info[0],
                                &hit_points[0],
                                hit_points.size(),
                                config.alpha,
                                config.photons_per_iter,
                                current_photon_radius2);*/
}


void CUDAEngine :: update_sample_buffer() {
  //kernels::cpu::update_sample_buffer_impl(&hit_points[0], hit_points.size(), config.width, sample_buffer);
}

void CUDAEngine :: splat_to_film() {
  //kernels::cpu::splat_to_film_impl(sample_buffer, film, config.width, config.height);
}

template<class T>
void cuda_alloc(T** ptr, unsigned num_elems) {
  cudaMalloc(ptr, sizeof(T) * num_elems);
}

void CUDAEngine :: before() {
  cudaFree(seeds);
  cudaFree(eye_paths);
  cudaFree(hit_points_info);
  cudaFree(hit_points);
  cudaFree(hash_grid);
  cudaFree(hash_grid_lengths);
  cudaFree(hash_grid_indexes);
  cudaFree(live_photon_paths);
}

void CUDAEngine :: after() {

  cuda_alloc(&seeds,            config.seed_size);
  cuda_alloc(&eye_paths,        config.total_hit_points);
  cuda_alloc(&hit_points_info,  config.total_hit_points);
  cuda_alloc(&hit_points,       config.total_hit_points);
  cuda_alloc(&hash_grid,         8 * config.total_hit_points);
  cuda_alloc(&hash_grid_lengths, config.total_hit_points);
  cuda_alloc(&hash_grid_indexes, config.total_hit_points);
  cuda_alloc(&live_photon_paths, config.photons_per_iter);
}


void CUDAEngine :: wait_for_all() {
}

}
