/*
 * contant.cuda.cu
 *
 *  Created on: Sep 16, 2012
 *      Author: jbarbosa
 */
#include <config/common.h>

__device__ __constant__ void* _dcache;
__device__ __constant__ unsigned int DeviceID_GPU;
__device__ __constant__ void* _memGPU;
__device__ __constant__ void* _cache;
__device__ __constant__ unsigned long* _pages;
__device__ __constant__ unsigned long* _pagesLock;
__device__ __constant__ unsigned long* _pagesCount;
