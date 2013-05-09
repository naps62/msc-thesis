#ifndef _PPM_KERNELS_GENERATE_EYE_PATHS_H_
#define _PPM_KERNELS_GENERATE_EYE_PATHS_H_

#include "utils/common.h"
#include <stddef.h>

namespace ppm { namespace kernels {

struct args_generate_eye_paths {
  const Config* config;
  PtrFreeScene* scene;
};

void k_generate_eye_paths (
  //void* eye_paths,   unsigned eye_path_count,    size_t eye_path_size,    // eye_path vector
  //void* seed_buffer, unsigned seed_buffer_count, size_t seed_buffer_size, // seed_buffer
  //const void* config, void* scene                                         // cl_args
  vector<EyePath> eye_paths,
  vector<Seed>    seed_buffer,
  const Config* config,
  PtrFreeScene*   scene
);

} }

#endif // _PPM_KERNELS_GENERATE_EYE_PATHS_H_
