/*
 * global.def.h
 *
 *  Created on: Sep 16, 2012
 *      Author: jbarbosa
 */

#ifndef GLOBAL_DEF_H_
#define GLOBAL_DEF_H_

#if defined(__CUDACC__)

extern __device__ __constant__ void* _dcache;
extern __device__ __constant__ unsigned int DeviceID_GPU;
extern __device__ __constant__ void* _memGPU;
extern __device__ __constant__ void* _cache;
extern __device__ __constant__ unsigned long* _pages;

#define DeviceID DeviceID_GPU
#define _gmem _memGPU

#else

extern __thread unsigned int DeviceID_CPU;
extern void* _memCPU;
#define DeviceID DeviceID_CPU
#define _gmem _memCPU

extern __thread void* _cache;
extern __thread unsigned long* _pages;

#endif

#endif /* GLOBAL_DEF_H_ */
