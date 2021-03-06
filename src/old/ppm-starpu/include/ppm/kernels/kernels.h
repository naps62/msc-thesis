#ifndef _PPM_KERNELS_H_
#define _PPM_KERNELS_H_

#include "utils/config.h"
#include "utils/random.h"
#include "ppm/ptrfreescene.h"
#include "ppm/types/paths.h"
#include "ppm/kernels/codelets.h"
#include <vector>
#include <starpu.h>
using std::vector;
using ppm::kernels::codelets::starpu_args;

namespace ppm { namespace kernels {

void generate_eye_paths (
    starpu_data_handle_t eye_paths,
    starpu_data_handle_t seed_buffer);

void generate_photon_paths (
    starpu_data_handle_t ray_buffer,
    starpu_data_handle_t photon_paths,
    starpu_data_handle_t seed_buffer
);
void advance_eye_paths (
    starpu_data_handle_t hit_points_info,
    starpu_data_handle_t hit_buffer,
    starpu_data_handle_t eye_paths,
    starpu_data_handle_t eye_paths_indexes,
    starpu_data_handle_t seed_buffer,
    const unsigned buffer_size
);

void advance_photon_paths (
    starpu_data_handle_t ray_buffer,
    starpu_data_handle_t hit_buffer,
    starpu_data_handle_t photon_paths,
    starpu_data_handle_t hit_points_info,
    starpu_data_handle_t hit_points,
    starpu_data_handle_t seed_buffer
);

void intersect_ray_hit_buffer (
    starpu_data_handle_t ray_buffer,
    starpu_data_handle_t hit_buffer,
    const unsigned buffer_size
);

void accum_flux (
    starpu_data_handle_t hit_points_info,
    starpu_data_handle_t hit_points,
    const unsigned photons_traced
);

} }

#endif // _PPM_KERNELS_H_
