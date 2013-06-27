#include "ppm/engine.h"
#include "ppm/kernels/codelets.h"
#include "utils/random.h"
#include "ppm/kernels/generate_eye_paths.h"
#include "ppm/kernels/intersect_ray_hit_buffer.h"
#include "ppm/kernels/advance_eye_paths.h"
#include "ppm/kernels/generate_photon_paths.h"
#include "ppm/kernels/advance_photon_paths.h"
#include "ppm/types.h"

#include <starpu.h>

namespace ppm {

//
// constructors
//

Engine :: Engine(const Config& _config)
: config(_config),
  scene(new PtrFreeScene(config)),
  hash_grid(config.total_hit_points),
  film(config),
  seeds(config.total_hit_points),
  hit_points_info(config.total_hit_points),
  hit_points(config.total_hit_points),
  chunk_size(config.engine_chunk_size) {

  // load display if necessary
  if (config.use_display) {
    display = new Display(config, film);
    display->start(true);
  }

  // load starpu
  starpu_conf_init(&this->spu_conf);
  spu_conf.sched_policy_name = config.sched_policy.c_str();

  starpu_init(&this->spu_conf);
  kernels::codelets::init();
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
  film.clear(Spectrum(1.f, 0.f, 0.f));
  cout << "Building seed buffer" << endl;
  this->init_seed_buffer();
  cout << "Building hit points" << endl;
  this->build_hit_points();
  cout << "hit points done" << endl;

  // init_radius
  this->init_radius();

  // set hash_grid data
  this->update_bbox();
  this->hash_grid.set_bbox(this->bbox);
  this->hash_grid.set_hit_points(hit_points_info, hit_points);

  // main loop
  unsigned iteration = 0;
  while(display->is_on() && iteration < config.max_iters) {

    // update lookup hash grid
    this->hash_grid.rehash();

    // advance photon patha
    this->advance_photon_paths();

    // accumulate flux

    // update frame buffer

    set_captions();
    display->request_update(config.min_frame_time);
  }
}

void Engine :: set_captions() {
  stringstream header, footer;
  header << "Hello World!";
  footer << "[Photons " << 0 << "M][Avg. photons/sec " << 0 << "K][Elapsed time " << 0 << "secs]";
  display->set_captions(header, footer);
}

//
// private methods
//

void Engine :: init_seed_buffer() {
  // TODO is it worth it to move this to a kernel?
  for(uint i = 0; i < config.total_hit_points; ++i) {
    seeds[i] = mwc(i);
  }
}

void Engine :: build_hit_points() {
  // list of eye paths to generate
  vector<EyePath> eye_paths(config.total_hit_points);

  // eye path generation
  cout << "generating eye paths" << endl;
  kernels::generate_eye_paths(eye_paths, seeds, &config, scene);

  cout << "eye paths to hit points" << endl;
  this->eye_paths_to_hit_points(eye_paths);
  cout << "success!" << endl;
}

void Engine :: eye_paths_to_hit_points(vector<EyePath>& eye_paths) {
  unsigned todo_eye_paths = eye_paths.size();
  const unsigned hit_points_count = hit_points_info.size();
  unsigned chunk_count = 0;
  unsigned chunk_done_count = 0;
  RayBuffer ray_hit_buffer(chunk_size);
  vector<unsigned> eye_paths_indexes(chunk_size);

  while (todo_eye_paths) {

    const unsigned start = chunk_count * chunk_size;
    const unsigned end   = (hit_points_count - start  < chunk_size) ? hit_points_count : start + chunk_size;

    // 1. fill the ray buffer
    for(unsigned i = start; i < end; ++i) {
      EyePath& eye_path = eye_paths[i];

      if (!eye_path.done) {
        // check if path reached max depth
        if (eye_path.depth > config.max_eye_path_depth) {
          // make it done
          HitPointStaticInfo& hp = hit_points_info[eye_path.sample_index];
          hp.type  = CONSTANT_COLOR;
          hp.scr_x = eye_path.scr_x;
          hp.scr_y = eye_path.scr_y;
          hp.throughput = Spectrum();

          eye_path.done = true;

        } else {
          // if not, add it to current buffer
          eye_path.depth++;
          const int index = ray_hit_buffer.AddRay(eye_path.ray);
          eye_paths_indexes[index] = i;
        }
      }

      // check if this ray is already done, but not yet splatted
      if (eye_path.done && !eye_path.splat) {
        eye_path.splat = true;
        todo_eye_paths--;
        chunk_done_count++;
        if (chunk_done_count == chunk_size) {
          // move to next chunka
          chunk_count++;
          chunk_done_count = 0;
        }
      }
    }

    // 2. advance ray buffer
    if (ray_hit_buffer.GetRayCount() > 0) {
      kernels::intersect_ray_hit_buffer(ray_hit_buffer, /*&config,*/ scene);
      kernels::advance_eye_paths(hit_points_info, ray_hit_buffer, eye_paths, eye_paths_indexes, seeds, /*&config,*/ scene);
      ray_hit_buffer.Reset();
    }
  }
}

void Engine :: init_radius() {
  BBox bbox = this->bbox;

  const Vector ssize = bbox.pMax - bbox.pMin;
  const float photon_radius = ((ssize.x + ssize.y + ssize.z) / 3.f) / ((config.width * config.spp + config.height * config.spp) / 2.f) * 2.f;
  const float photon_radius2 = photon_radius * photon_radius;

  bbox.Expand(photon_radius);
  for(unsigned i = 0; i < hit_points.size(); ++i) {
    hit_points[i].accum_photon_radius2 = photon_radius2;
  }
}

void Engine :: advance_photon_paths() {
  std::vector<PhotonPath> live_photon_paths(chunk_size);
  RayBuffer ray_hit_buffer(chunk_size);

  kernels::generate_photon_paths(ray_hit_buffer, live_photon_paths, seeds, &config, scene);

  kernels::advance_photon_paths(ray_hit_buffer, live_photon_paths, seeds, &config, scene);
}

void Engine :: update_bbox() {
  BBox bbox;

  // TODO move this to a kernel?
  for(unsigned i = 0; i < hit_points.size(); ++i) {
    HitPointStaticInfo& hpi = hit_points_info[i];
    if (hpi.type == SURFACE)
      bbox = Union(bbox, hpi.position);
  }
  this->bbox = bbox;
}

}
