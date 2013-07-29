#ifndef _PPM_KERNELS_CPU_H_
#define _PPM_KERNELS_CPU_H_

namespace ppm { namespace kernels {

  namespace cpu {
    typedef void cpu_kernel(void* buffers[], void* args);

    cpu_kernel generate_eye_paths;
    cpu_kernel advance_eye_paths;
    cpu_kernel bbox_zero_initialize;
    cpu_kernel bbox_reduce;
    cpu_kernel rehash;
    cpu_kernel generate_photon_paths;
    cpu_kernel advance_photon_paths;
    cpu_kernel accum_flux;

  }
}}

#endif // _PPM_KERNELS_CPU_H_
