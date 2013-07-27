#include "ppm/engine.h"
#include "ppm/kernels/codelets.h"
#include "utils/random.h"
#include "ppm/kernels/kernels.h"
#include "ppm/types.h"

#include <starpu.h>

namespace ppm {

//
// constructors
//

Engine :: Engine(const Config& _config, unsigned worker_count)
: iteration(1),
  config(_config),
  scene(new PtrFreeScene(config)),
  hash_grid(config.total_hit_points  /* TODO why is this? */),
  film(config.width, config.height),
  chunk_size(config.engine_chunk_size),

  seeds(config.total_hit_points),
  eye_paths(config.total_hit_points),
  eye_paths_indexes(chunk_size),
  ray_hit_buffer(chunk_size),
  hit_points_info(config.total_hit_points),
  hit_points(config.total_hit_points),
  live_photon_paths(chunk_size),

  sample_buffer(config.total_hit_points),
  sample_frame_buffer(config.width, config.height) {

  sample_frame_buffer.Clear();
  film.Reset();

  // load display if necessary
  if (config.use_display) {
    display = new Display(config, film);
    display->start(true);
  }

  // load starpu
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
  //this->build_hit_points();
  //this->update_bbox();
  //this->init_radius();

  this->hash_grid.set_bbox(this->bbox);
  this->hash_grid.set_hit_points(hit_points_info, hit_points);
  // main loop
  unsigned photons_traced = 0;
  while((!display || display->is_on()) && iteration < config.max_iters) {
    this->build_hit_points();
    this->update_bbox();
    this->init_radius();


    this->hash_grid.set_bbox(this->bbox);
    this->hash_grid.rehash(current_photon_radius2);

    kernels::generate_photon_paths(ray_buffer_h, live_photon_paths_h, seeds_h);
    kernels::advance_photon_paths(ray_buffer_h, hit_buffer_h, live_photon_paths_h, hit_points_info_h, hit_points_h, seeds_h, current_photon_radius2);

    photons_traced += chunk_size;

    for(unsigned i = 0; i < config.total_hit_points; ++i) {
      HitPointPosition& hpi = hit_points_info[i];
      HitPointRadiance& hp = hit_points[i];
      std::cout << i << " " << hp.accum_reflected_flux << '\n';
    }
    std::cout << "\n\n";
    exit(0);
    kernels::accum_flux(hit_points_info_h, hit_points_h, photons_traced, current_photon_radius2);

    //cout << '\n';
    //if (iteration == 1) exit(0);
    this->update_sample_frame_buffer();

    set_captions();
    display->request_update(config.min_frame_time);

    iteration++;
  }
}

void Engine :: set_captions() {
  stringstream header, footer;
  header << "Hello World!";
  footer << "[Photons " << rand() << "M][Avg. photons/sec " << 0 << "K][Elapsed time " << 0 << "secs]";
  display->set_captions(header, footer);
}

//
// private methods
//

void Engine :: init_starpu_handles() {
  starpu_vector_data_register(&seeds_h,             0, (uintptr_t)&seeds[0],                     seeds.size(),             sizeof(Seed));
  starpu_vector_data_register(&eye_paths_h,         0, (uintptr_t)&eye_paths[0],                 eye_paths.size(),         sizeof(EyePath));
  starpu_vector_data_register(&eye_paths_indexes_h, 0, (uintptr_t)&eye_paths_indexes[0],         eye_paths_indexes.size(), sizeof(unsigned));
  starpu_vector_data_register(&ray_buffer_h,        0, (uintptr_t)ray_hit_buffer.GetRayBuffer(), ray_hit_buffer.GetSize(), sizeof(Ray));
  starpu_vector_data_register(&hit_buffer_h,        0, (uintptr_t)ray_hit_buffer.GetHitBuffer(), ray_hit_buffer.GetSize(), sizeof(RayHit));
  starpu_vector_data_register(&hit_points_info_h,   0, (uintptr_t)&hit_points_info[0],           hit_points_info.size(),   sizeof(HitPointPosition));
  starpu_vector_data_register(&hit_points_h,        0, (uintptr_t)&hit_points[0],                hit_points.size(),        sizeof(HitPointPosition));
  starpu_vector_data_register(&live_photon_paths_h, 0, (uintptr_t)&live_photon_paths[0],         live_photon_paths.size(), sizeof(PhotonPath));
}

void Engine :: init_seed_buffer() {
  for(uint i = 0; i < config.total_hit_points; ++i) {
    seeds[i] = mwc(i+100);
  }
}

void Engine :: build_hit_points() {
  kernels::generate_eye_paths(eye_paths_h, seeds_h);
  this->eye_paths_to_hit_points();
}

void Engine :: eye_paths_to_hit_points() {
  int todo_eye_paths = eye_paths.size();
  const unsigned hit_points_count = hit_points_info.size();
  unsigned chunk_count = 0;
  unsigned chunk_finished = 0;

  while (todo_eye_paths > 0) {

    const unsigned start = chunk_count * chunk_size;
    const unsigned end   = (hit_points_count - start  < chunk_size) ? hit_points_count : start + chunk_size;

    unsigned current_buffer_size = 0;

    for(unsigned i = start; i < end; ++i) {
      EyePath& eye_path = eye_paths[i];

      if (!eye_path.done) {
        // check if path reached max depth
        if (eye_path.depth > config.max_eye_path_depth) {
          // make it done
          HitPointPosition& hp = hit_points_info[eye_path.sample_index];
          hp.type = CONSTANT_COLOR;
          hp.scr_x = eye_path.scr_x;
          hp.scr_y = eye_path.scr_y;
          hp.throughput = Spectrum();

          eye_path.done = true;
        } else {
          eye_path.depth++;
          ray_hit_buffer.GetRayBuffer()[current_buffer_size] = eye_path.ray;
          eye_paths_indexes[current_buffer_size] = i;
          current_buffer_size++;
        }
      } else {
      }

      if (eye_path.done && !eye_path.splat) {
        --todo_eye_paths;
        eye_path.splat = true;
        chunk_finished++;
        if (chunk_finished == chunk_size) {
          chunk_count++;
          chunk_finished = 0;
        }
      }
    }

    if (current_buffer_size > 0) {
      kernels::intersect_ray_hit_buffer(ray_buffer_h, hit_buffer_h, current_buffer_size);
      kernels::advance_eye_paths(hit_points_info_h, hit_buffer_h, eye_paths_h, eye_paths_indexes_h, seeds_h, current_buffer_size);
    }
  }
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
  bbox.Expand(photon_radius);
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
    HitPointPosition& hpi = hit_points_info[i];
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
