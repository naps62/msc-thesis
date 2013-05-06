#include "ppm/kernels/build_hit_points.h"

#include <stdio.h>

_extern_c_ void k_cpu_build_hit_points(void* buffers[], void* args) {
  printf("k_cpu_build_hit_points\n");
}
