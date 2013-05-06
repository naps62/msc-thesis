#include "ppm/kernels/build_hit_points.h"

#include <stdio.h>

static __global__ void k_cuda_build_hit_points_impl() {

}

_extern_c_ void k_cuda_build_hit_points(void* buffers[], void* args) {
  printf("k_cuda_build_hit_points\n");
}
