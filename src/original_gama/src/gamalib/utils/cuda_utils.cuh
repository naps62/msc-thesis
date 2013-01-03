/*
 * cuda_utils.h
 *
 *  Created on: Apr 2, 2012
 *      Author: jbarbosa
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#ifdef __CUDACC__

__inline__ __device__ unsigned int get_smid(void) {
	unsigned int ret = 0;
	asm("mov.u32 %0, %smid;" : "=r"(ret) );
	return ret;
}

__inline__ __device__ unsigned int get_warpid(void) {
	unsigned int ret = 0;
	asm("mov.u32 %0, %warpid;" : "=r"(ret) );
	__threadfence();
	return ret;
}

template<typename T>
__inline__ __device__ void memcpy_SIMD(T* dst, T* src, int nElement ) {

	for(int i=0; i+threadIdx.x < nElement/sizeof(T*); i+= 32) {
		dst[i+threadIdx.x] = src[i+threadIdx.x];
	}
	__threadfence_system();
	return;
}

#endif

#endif /* CUDA_UTILS_H_ */
