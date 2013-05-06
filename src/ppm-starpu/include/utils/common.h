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


#if defined(__cplusplus)
#define _extern_c_ extern "C"

#else
#define _extern_c_
#endif

#endif // _UTILS_COMMON_H
