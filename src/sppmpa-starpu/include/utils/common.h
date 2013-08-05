#ifndef _UTILS_COMMON_H
#define _UTILS_COMMON_H

#if defined(__CUDACC__)
#define __HYBRID__ __host__ __device__
#define __HD__ __HYBRID__
#define __forceinline __forceinline__

#else
#define __HYBRID__
#define __HD__
#define __forceinline __inline__ __attribute__((__always_inline__))
#endif

#define CUDA_SAFE(exp) { \
  cudaError_t ret = (exp); \
  if (cudaSuccess != ret) { \
    printf("Error %d in %s:%d\n", ret, __FILE__, __LINE__); \
    exit(-1); \
  } \
}

namespace ppm {
  template<class T>
  void delete_array(T*& arr) {
    if (arr) {
      delete[] arr;
      arr = 0;
    }
  }

  template<class T>
  void reset_array(T*& arr, unsigned new_size) {
    delete_array(arr);
    arr = new T[new_size];
  }
}

#endif // _UTILS_COMMON_H
