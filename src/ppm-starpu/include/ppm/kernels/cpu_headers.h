#ifndef _PPM_KERNELS_CPU_H_
#define _PPM_KERNELS_CPU_H_

namespace ppm { namespace kernels {

  namespace cpu {
    typedef void cpu_kernel(void* buffers[], void* args);

    //void generate_eye_paths(void* buffers[], void* args);
    //void eye_paths_to_hit_points(void *buffers)
    cpu_kernel generate_eye_paths;
    cpu_kernel eye_paths_to_hit_points;
  }

}}

#endif // _PPM_KERNELS_CPU_H_
