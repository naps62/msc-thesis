#ifndef _PPM_KERNELS_CPU_H_
#define _PPM_KERNELS_CPU_H_

namespace ppm { namespace kernels {

  namespace cpu {
    typedef void cpu_kernel(void* buffers[], void* args);

    cpu_kernel generate_eye_paths;
    cpu_kernel intersect_ray_hit_buffer;
    cpu_kernel advance_eye_paths;
  }

}}

#endif // _PPM_KERNELS_CPU_H_
