/*
 * macros.def.h
 *
 *  Created on: Sep 16, 2012
 *      Author: jbarbosa
 */

#ifndef MACROS_DEF_H_
#define MACROS_DEF_H_

#if defined(__CUDACC__)

#define __HYBRID__ __host__ __device__
#define __DEVICE__ __device__

#define __forceinline 	
//__forceinline__
#define __noinline 		__noinline__

#define SIMDserial for (int __SerialCode=0; __SerialCode < NTHREAD ; __SerialCode++) if(__SerialCode == (threadIdx.x & 31))
#define SingleThread __shared__ int __X__; __X__ = threadIdx.x; if ( __X__ == threadIdx.x )
#define TID (blockIdx.x * blockDim.x + threadIdx.x)
#define TID_SIZE (blockDim.x * gridDim.x)

#define __gshared__ __shared__
#define CORE_ID (DeviceID << 16) | get_smid()


#else


#define __HYBRID__
#define __DEVICE__

#define __forceinline
//__inline__ __attribute__((__always_inline__))
#define __noinline

#define SIMDserial
#define SingleThread

#define atomicCAS(lock,valC,valS) __sync_val_compare_and_swap((volatile unsigned int*)lock,valC,valS)
#define atomicAdd(var,valC) __sync_fetch_and_add((volatile unsigned int*)var,valC)
#define atomicDec(var,valC) __sync_fetch_and_sub((volatile unsigned int*)var,valC)
#define atomicExch(var,valC) (*var) = valC
#define __all
#define __any

#define TID 0
#define TID_SIZE 1

#define CORE_ID (DeviceID << 16)

#define __gshared__
#endif

#define forall(__Dom, __it) for(int __it = __Dom.sx._d0; __it < __Dom.sx._d1; __it++)
#define forallptr(__Dom, __it) for(int __it = __Dom->sx._d0; __it < __Dom->sx._d1; __it++)
//#define forallsimd(__Dom, __it, __wide) for(int __it = __Dom->sx._d0; __it < __Dom->sx._d1; __it+=__wide)

#define forallSIMD(__lo, __hi, __wide) for(; __lo < __hi; __lo+=__wide)



#endif /* MACROS_DEF_H_ */
