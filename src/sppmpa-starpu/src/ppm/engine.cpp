#include "ppm/engine.h"
#include "ppm/kernels/codelets.h"
#include "utils/random.h"
#include "ppm/kernels/codelets.h"
#include "ppm/kernels/kernels.h"
#include "ppm/types.h"

#include <starpu.h>
using namespace ppm::kernels;

namespace ppm {

//
// constructors
//

Engine :: Engine(const Config& _config, unsigned worker_count)
: iteration(1),
  total_photons_traced(0),
  config(_config),
  scene(new PtrFreeScene(config)),
  hash_grid(config.total_hit_points /* TODO why is this? */),
  film(config.width, config.height),
  chunk_size(config.engine_chunk_size),

  seeds(max(config.total_hit_points, chunk_size)),
  eye_paths(config.total_hit_points),
  hit_points_info(config.total_hit_points),
  hit_points(config.total_hit_points),
  live_photon_paths(chunk_size),

  sample_buffer(config.width * config.height * config.spp * config.spp),
  sample_frame_buffer(config.width, config.height) {

  sample_frame_buffer.Clear();
  film.Reset();

  // load display if necessary
  if (config.use_display) {
    display = new Display(config, film);
    display->start(true);
  }

  // load starpuk
  starpu_conf_init(&this->spu_conf);
  spu_conf.sched_policy_name = config.sched_policy.c_str();

  starpu_init(&this->spu_conf);
  kernels::codelets::init(&config, scene, &hash_grid, NULL, NULL, NULL); // TODO GPU versions here

  init_seed_buffer();
  init_starpu_handles();
}

Engine :: ~Engine() {
  // wait for display to close
  if (config.use_display) {
    display->join();
  }

  starpu_shutdown();
}

//
// public methods
//
void Engine :: render() {
  start_time = WallClockTime();

  this->hash_grid.set_hit_points(hit_points_info, hit_points);

  // main loop
  while((!display || display->is_on()) && iteration <= config.max_iters) {
    starpu_insert_task(&codelets::generate_eye_paths,
      STARPU_RW, eye_paths_h,
      STARPU_RW, seeds_h,
      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
      0);

    starpu_insert_task(&codelets::advance_eye_paths,
      STARPU_RW, hit_points_info_h,
      STARPU_RW, eye_paths_h,
      STARPU_RW, seeds_h,
      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
      0);
    starpu_task_wait_for_all(); // TODO remove this from here later

    this->update_bbox();
    this->init_radius();
    this->hash_grid.set_bbox(this->bbox);
    this->hash_grid.rehash(current_photon_radius2);

    starpu_insert_task(&codelets::generate_photon_paths,
      STARPU_RW, live_photon_paths_h,
      STARPU_RW, seeds_h,
      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
      0);

    starpu_insert_task(&codelets::advance_photon_paths,
      STARPU_RW, live_photon_paths_h,
      STARPU_R,  hit_points_info_h,
      STARPU_RW, hit_points_h,
      STARPU_RW, seeds_h,
      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
      STARPU_VALUE, &current_photon_radius2, sizeof(current_photon_radius2),
      0);


    starpu_insert_task(&codelets::accum_flux,
      STARPU_R,  hit_points_info_h,
      STARPU_RW, hit_points_h,
      STARPU_VALUE, &codelets::generic_args, sizeof(codelets::generic_args),
      STARPU_VALUE, &chunk_size,             sizeof(chunk_size),
      STARPU_VALUE, &current_photon_radius2, sizeof(current_photon_radius2),
      0);
    starpu_task_wait_for_all(); // TODO remove this from here later

    this->update_sample_frame_buffer();

    total_photons_traced += chunk_size;
    iteration++;

    if (display) {
      set_captions();
      display->request_update(config.min_frame_time);
    }
  }
}

void Engine :: set_captions() {
  const double elapsed_time = WallClockTime() - start_time;
  const unsigned long long total_photons_M = float(total_photons_traced / 1000000.f);
  const unsigned long long photons_per_sec = total_photons_traced / (elapsed_time * 1000.f);

  stringstream header, footer;
  header << "Hello World!";
  footer << std::setprecision(2) << "[Photons " << total_photons_M << "M][Avg. photons/sec " << photons_per_sec << "K][Elapsed time " << int(elapsed_time) << "secs]";
  display->set_captions(header, footer);
}

//
// private methods
//

void Engine :: init_starpu_handles() {
  starpu_vector_data_register(&seeds_h,             0, (uintptr_t)&seeds[0],             seeds.size(),             sizeof(Seed));
  starpu_vector_data_register(&eye_paths_h,         0, (uintptr_t)&eye_paths[0],         eye_paths.size(),         sizeof(EyePath));
  starpu_vector_data_register(&hit_points_info_h,   0, (uintptr_t)&hit_points_info[0],   hit_points_info.size(),   sizeof(HitPointPosition));
  starpu_vector_data_register(&hit_points_h,        0, (uintptr_t)&hit_points[0],        hit_points.size(),        sizeof(HitPointPosition));
  starpu_vector_data_register(&live_photon_paths_h, 0, (uintptr_t)&live_photon_paths[0], live_photon_paths.size(), sizeof(PhotonPath));
}

void Engine :: init_seed_buffer() {
  for(uint i = 0; i < seeds.size(); ++i) {
    seeds[i] = mwc(i+100);
  }
}

void Engine :: build_hit_points() {
}

void Engine :: init_radius() {
  const Vector ssize = bbox.pMax - bbox.pMin;
  const float photon_radius = ((ssize.x + ssize.y + ssize.z) / 3.f) / ((config.width * config.spp + config.height * config.spp) / 2.f) * 2.f;
  current_photon_radius2 = photon_radius * photon_radius;

  float g = 1;
  for(uint k = 1; k < iteration; ++k)
    g *= (k + config.alpha) / k;

  g /= iteration;
  current_photon_radius2 *= g;
  bbox.Expand(sqrt(current_photon_radius2));
}

void Engine :: update_bbox() {
  BBox bbox;

  // TODO move this to a kernel?
  for(unsigned i = 0; i < hit_points.size(); ++i) {
    HitPointPosition& hpi = hit_points_info[i];
    if (hpi.type == SURFACE) {
      bbox = Union(bbox, hpi.position);
    }
  }
  this->bbox = bbox;
}

void Engine :: update_sample_frame_buffer() {
  for(unsigned i = 0; i < hit_points.size(); ++i) {
    HitPointRadiance& hp = hit_points[i];

    const float scr_x = i % config.width;
    const float scr_y = i / config.width;

    sample_buffer.SplatSample(scr_x, scr_y, hp.radiance);
  }

  sample_frame_buffer.Clear();
  if (sample_buffer.GetSampleCount() > 0) {
    film.SplatSampleBuffer(&sample_frame_buffer, true, &sample_buffer);
    sample_buffer.Reset();
  }
}

}
