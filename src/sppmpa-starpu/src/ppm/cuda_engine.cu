#include "ppm/cpu_engine.h"
#include "ppm/cuda_engine.h"
#include "ppm/kernels/helpers.cuh"
#include "ppm/kernels/codelets.h"
using namespace ppm::kernels;

namespace ppm {

//
// constructors
//

CUDAEngine :: CUDAEngine(const Config& _config)
: Engine(_config),
  cuda_scene(scene->to_device(0)) {

  cudaStreamCreate(&stream);
}

CUDAEngine :: ~CUDAEngine() {
}


//
// private methods
//

void CUDAEngine :: init_seed_buffer() {
  const double start_time = WallClockTime();
  const unsigned size = config.seed_size;
  const unsigned threads_per_block = config.cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  kernels::cuda::init_seeds_impl
  <<<n_blocks, threads_per_block, 0, stream>>>
   (seeds,
    size,
    iteration);
}

void CUDAEngine :: generate_eye_paths() {
  const double start_time = WallClockTime();
  const unsigned width = config.width;
  const unsigned height = config.height;
  const unsigned block_side = config.cuda_block_size_sqrt;
  const dim3 threads_per_block = dim3(block_side,                block_side);
  const dim3 n_blocks          = dim3(std::ceil(width/(float)threads_per_block.x), std::ceil(height/(float)threads_per_block.y));

  kernels::cuda::generate_eye_paths_impl
  <<<n_blocks, threads_per_block, 0, stream>>>
   (eye_paths,
    seeds,
    width,
    height,
    cuda_scene);
}

void CUDAEngine :: advance_eye_paths() {
  const double start_time = WallClockTime();
  const unsigned size = config.total_hit_points;
  const unsigned threads_per_block = config.cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  kernels::cuda::advance_eye_paths_impl
  <<<n_blocks, threads_per_block, 0, stream>>>
   (hit_points_info,
    eye_paths,
    size,
    seeds,
    cuda_scene,
    config.max_eye_path_depth);
}

void CUDAEngine :: bbox_compute() {
  const double start_time = WallClockTime();
  const unsigned total_spp = config.width * config.spp + config.height * config.spp;

  cudaMemcpyAsync(host_hit_points_info, hit_points_info, sizeof(HitPointPosition)*config.total_hit_points, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  kernels::cpu::bbox_compute_impl(host_hit_points_info,
                                  config.total_hit_points,
                                  *host_bbox,
                                  *host_current_photon_radius2,
                                  iteration,
                                  total_spp,
                                  config.alpha);
  cudaMemcpyAsync(bbox, host_bbox, sizeof(BBox), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(current_photon_radius2, host_current_photon_radius2, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void CUDAEngine :: rehash() {
  const double start_time = WallClockTime();
  kernels::cpu::rehash_impl(host_hit_points_info,
                            config.total_hit_points,
                            host_hash_grid,
                            host_hash_grid_lengths,
                            host_hash_grid_indexes,
                            host_inv_cell_size,
                            *host_bbox,
                            *host_current_photon_radius2);
  cudaMemcpyAsync(hash_grid,         host_hash_grid,         sizeof(unsigned) * 8 * config.total_hit_points, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(hash_grid_indexes, host_hash_grid_indexes, sizeof(unsigned) *     config.total_hit_points, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(hash_grid_lengths, host_hash_grid_lengths, sizeof(unsigned) *     config.total_hit_points, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(inv_cell_size,     host_inv_cell_size,     sizeof(float),                                  cudaMemcpyHostToDevice, stream);
}

void CUDAEngine :: generate_photon_paths() {
  const double start_time = WallClockTime();
  const unsigned size = config.photons_per_iter;
  const unsigned threads_per_block = config.cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  kernels::cuda::generate_photon_paths_impl
  <<<n_blocks, threads_per_block, 0, stream>>>
   (live_photon_paths,
    size,
    seeds,
    cuda_scene);
}

void CUDAEngine :: advance_photon_paths() {
  const double start_time = WallClockTime();
  const unsigned size = config.photons_per_iter;
  const unsigned threads_per_block = config.cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  kernels::cuda::advance_photon_paths_impl
  <<<n_blocks, threads_per_block, 0, stream>>>
   (live_photon_paths,
    size,
    seeds,
    cuda_scene,
    hit_points_info,
    hit_points,
    bbox,
    config.max_photon_depth,
    current_photon_radius2,
    config.total_hit_points,

    hash_grid,
    hash_grid_lengths,
    hash_grid_indexes,
    inv_cell_size);
}

void CUDAEngine :: accumulate_flux() {
  const double start_time = WallClockTime();
  const unsigned size = config.total_hit_points;
  const unsigned threads_per_block = config.cuda_block_size;
  const unsigned n_blocks          = std::ceil(size / (float)threads_per_block);

  kernels::cuda::accum_flux_impl
  <<<n_blocks, threads_per_block, 0, stream>>>
   (hit_points_info,
    hit_points,
    size,
    config.alpha,
    config.photons_per_iter,
    current_photon_radius2);
}


void CUDAEngine :: update_sample_buffer() {
  const double start_time = WallClockTime();
  cudaMemcpyAsync(host_hit_points, hit_points, sizeof(HitPointRadiance)*config.total_hit_points, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  kernels::cpu::update_sample_buffer_impl(host_hit_points, config.total_hit_points, config.width, sample_buffer, config.max_threads);
}

void CUDAEngine :: splat_to_film() {
  kernels::cpu::splat_to_film_impl(sample_buffer, film, config.width, config.height);
}

template<class T>
void cuda_alloc(T** ptr, unsigned num_elems) {
  cudaMalloc(ptr, sizeof(T) * num_elems);
  cudaMemset(*ptr, 0, sizeof(T) * num_elems);
}

void CUDAEngine :: before() {

  cuda_alloc(&seeds,            config.seed_size);
  cuda_alloc(&eye_paths,        config.total_hit_points);
  cuda_alloc(&hit_points_info,  config.total_hit_points);
  cuda_alloc(&hit_points,       config.total_hit_points);
  cuda_alloc(&hash_grid,         8 * config.total_hit_points);
  cuda_alloc(&hash_grid_lengths, config.total_hit_points);
  cuda_alloc(&hash_grid_indexes, config.total_hit_points);
  cuda_alloc(&live_photon_paths, config.photons_per_iter);
  cuda_alloc(&bbox,                   1);
  cuda_alloc(&inv_cell_size,          1);
  cuda_alloc(&current_photon_radius2, 1);

  host_hit_points_info        = new HitPointPosition[config.total_hit_points];
  host_hit_points             = new HitPointRadiance[config.total_hit_points];
  host_hash_grid              = new unsigned[8 * config.total_hit_points];
  host_hash_grid_indexes      = new unsigned[config.total_hit_points];
  host_hash_grid_lengths      = new unsigned[config.total_hit_points];
  host_bbox                   = new BBox();
  host_inv_cell_size          = new float;
  host_current_photon_radius2 = new float;

  host_seeds = new Seed[config.seed_size];
  host_eye_paths = new EyePath[config.total_hit_points];
  host_photon_paths = new PhotonPath[config.photons_per_iter];
}

void CUDAEngine :: after() {
  cudaFree(seeds);
  cudaFree(eye_paths);
  cudaFree(hit_points_info);
  cudaFree(hit_points);
  cudaFree(hash_grid);
  cudaFree(hash_grid_lengths);
  cudaFree(hash_grid_indexes);
  cudaFree(live_photon_paths);
  cudaFree(bbox);
  cudaFree(inv_cell_size);
  cudaFree(current_photon_radius2);

  delete[] host_hit_points_info;
  delete[] host_hit_points;
  delete[] host_hash_grid;
  delete[] host_hash_grid_indexes;
  delete[] host_hash_grid_lengths;
  delete[] host_bbox;
  delete[] host_inv_cell_size;
  delete[] host_current_photon_radius2;

  delete[] host_seeds;
  delete[] host_eye_paths;
  delete[] host_photon_paths;
}


void CUDAEngine :: wait_for_all() {
  cudaStreamSynchronize(stream);
}

}
