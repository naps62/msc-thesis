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

  // starpu stuff
  starpu_conf spu_conf;

  SampleBuffer* sample_buffer;
  SampleFrameBuffer* frame_buffer;

  std::vector<Seed> seeds;
  std::vector<EyePath> eye_paths;
  std::vector<HitPointPosition> hit_points_info;
  std::vector<HitPointRadiance> hit_points;
  std::vector<PhotonPath> live_photon_paths;


  void render();
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
};

}

#endif // _PPM_CPU_ENGINE_H_
