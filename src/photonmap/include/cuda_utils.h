/*
 * cuda_utils.h
 *
 *  Created on: Apr 13, 2012
 *      Author: rr
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include "cuda.h"
#include "cuda_runtime.h"

#define LANE0 ((threadIdx.x & 31)==0)    // or warp lane index
#if !defined __DEBUG && !defined __RELEASE
#warning "Debug or release flags not specified"
#endif
//Round a / b to nearest higher integer value
static int IntDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
static void ComputeGridSize(int n, int blockSize, int &numBlocks, int &numThreads) {
	numThreads = min(blockSize, n);
	numBlocks = IntDivUp(n, numThreads);
	assert(numBlocks < 65535);
}

void inline checkCUDAmemory(char* t = NULL) {
//#warning checking memory forced
#ifdef __DEBUG
	cudaDeviceSynchronize();
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	fprintf(stderr, "%s mem %ld total %ld\n", t, free / 1024 / 1024,
			total / 1024 / 1024);
#endif
}

void inline checkCUDAError(char* t = NULL) {
//#warning checking error forced
#ifdef __DEBUG
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error %s: %s.\n", t, cudaGetErrorString(err));
		exit(-1);
	}
#endif
}

void inline __E(cudaError_t err) {
#ifdef __DEBUG

	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA Runtime API error: %s.\n",
				cudaGetErrorString(err));
		exit(-1);
	}
#endif

}

#endif /* CUDA_UTILS_H_ */
