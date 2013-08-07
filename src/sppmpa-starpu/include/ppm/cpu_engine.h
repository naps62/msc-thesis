#ifndef _PPM_CPU_ENGINE_H_
#define _PPM_CPU_ENGINE_H_

#include "ppm/engine.h"

namespace ppm {

class CPUEngine : public Engine {
public:
  CPUEngine(const Config& _config);
  ~CPUEngine();

protected:
  BBox bbox;
  float current_photon_radius2;

  std::vector<Seed> seeds;
  std::vector<EyePath> eye_paths;
  std::vector<HitPointPosition> hit_points_info;
  std::vector<HitPointRadiance> hit_points;
  std::vector<PhotonPath> live_photon_paths;

  std::vector<unsigned> hash_grid;
  std::vector<unsigned> hash_grid_indexes;
  std::vector<unsigned> hash_grid_lengths;
  float inv_cell_size;


  void output();

  void init_seed_buffer();
  void generate_eye_paths();
  void advance_eye_paths();
  void bbox_compute();
  void rehash();
  void generate_photon_paths();
  void advance_photon_paths();
  void accumulate_flux();
  void update_sample_buffer();
  void splat_to_film();

  void wait_for_all();
  void before();
  void after();
};


namespace kernels { namespace cpu {
  void init_seeds_impl(Seed* const seeds, const unsigned size, const unsigned iteration);
  void generate_eye_paths_impl(EyePath* const eye_paths, Seed* const seed_buffer, const unsigned width, const unsigned height, const PtrFreeScene* scene);
  void advance_eye_paths_impl(HitPointPosition* const hit_points, EyePath* const eye_paths, const unsigned eye_paths_count, Seed* const seed_buffer, const PtrFreeScene* const scene, const unsigned max_eye_path_depth);
  void bbox_compute_impl(const HitPointPosition* const points, const unsigned size, BBox& bbox, float& photon_radius2, const float iteration, const float total_spp, const float alpha);

  void rehash_impl(
    const HitPointPosition* const hit_points_info, unsigned size,
    unsigned*  hash_grid,
    unsigned*  hash_grid_lengths,
    unsigned*  hash_grid_indexes,
    float* inv_cell_size,
    const BBox& bbox,
    const float current_photon_radius2);

  void generate_photon_paths_impl(
      PhotonPath* const photon_paths, const unsigned photon_paths_count,
      Seed* const seed_buffer,        // const unsigned seed_buffer_count,
      const PtrFreeScene* scene);

  void advance_photon_paths_impl(
      PhotonPath* const photon_paths,    const unsigned photon_paths_count,
      Seed* const seed_buffer,        // const unsigned seed_buffer_count,
      const PtrFreeScene* scene,

      HitPointPosition* const hit_points_info,
      HitPointRadiance* const hit_points,
      const BBox& bbox,
      const unsigned CONST_max_photon_depth,
      const float photon_radius2,
      const unsigned hit_points_count,

      const unsigned*           hash_grid,
      const unsigned*           hash_grid_lengths,
      const unsigned*           hash_grid_indexes,
      const float               hash_grid_inv_cell_size);

  void accum_flux_impl(
    const HitPointPosition* const hit_points_info,
    HitPointRadiance* const hit_points,
    const unsigned size,
    const float alpha,
    const unsigned photons_traced,
    const float current_photon_radius2);

  void update_sample_buffer_impl(
    const HitPointRadiance* const hit_points,
    const unsigned size,
    const unsigned width,
    SampleBuffer* const buffer);

  void splat_to_film_impl(
    luxrays::SampleBuffer* const buffer,
    Film* const film,
    const unsigned width,
    const unsigned height);
} }

}

#endif // _PPM_CPU_ENGINE_H_
