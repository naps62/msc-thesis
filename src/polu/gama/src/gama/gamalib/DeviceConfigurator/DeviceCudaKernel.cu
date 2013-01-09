/*
 * DeviceCuda.cu
 *
 *  Created on: Apr 11, 2012
 *      Author: jbarbosa
 */

#include <cuda.h>

#include <config/common.h>

#include <gamalib/memlib/LowLevelMemAllocator.h>

#include "DeviceCuda.h"

#include <config/vtable.cuh>
#include <gamalib/utils/cuda_utils.cuh>
#include <gamalib/GenericKernels/KernelCuda.cuh>


#include <gamalib/utils/x86_utils.h>

#define PostWork(W) outbox->enqueue(W);

typedef unsigned char byte;

extern "C++" {
void CallBackStartSampling(CUstream st, cudaError_t error, void* userData);
void CallBackStopSampling(CUstream st, cudaError_t error, void* userData);
void CallBackGeneric(CUstream st, cudaError_t error, void* userData);
void CallBackWide(CUstream st, cudaError_t error, void* userData);
}

__global__ void
genericKernel(Workqueue<work, INBOX_QUEUE_SIZE, GPU_CUDA>* INBOX ) {

	unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
	work* w_item;

	if( (w_item = INBOX->data[index]) != NULL) {
		(w_item->*WORK_GPU_TABLE[w_item->getWorkTypeID()])();
	}

}


__global__ 
void genericWideKernel(work* w_item) {
    work* w;
    w = w_item;
	(w->*WORK_GPU_TABLE[w->getWorkTypeID()])();
}

__inline__ void CudaTest(char *msg) {
	cudaError_t e;

	cudaThreadSynchronize();
	if (cudaSuccess != (e = cudaGetLastError())) {
		fprintf(stderr, "%s: %d\n", msg, e);
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		exit(-1);
	}
}
void DeviceCuda::classWideKernel(Information * SIGNAL) {
	checkCudaErrors(cudaSetDevice(cudaDeviceID)); // Sem isto estoura!!!! LOL
	unsigned int st = __sync_fetch_and_add((volatile unsigned int*)&stream,1) & 31;

#ifdef __SYNC
	cudaEvent_t syncPoint;
	checkCudaErrors(cudaEventCreate(&syncPoint));
#endif

#if defined GAMA_CACHE
	std::vector<pointerInfo>* Lw = (SIGNAL->Work->*TOCACHEW_CPU_TABLE[SIGNAL->Work->getWorkTypeID()])();
	std::vector<pointerInfo>* Lr = (SIGNAL->Work->*TOCACHER_CPU_TABLE[SIGNAL->Work->getWorkTypeID()])();

	for(int i=0; i< Lw->size(); i++){
		cache->cachePtr((*Lw)[i].ptr,(*Lw)[i].lenght,&streams[st],CACHE_READ_WRITE);
	}

	for(int i=0; i< Lr->size(); i++) {
		cache->cachePtr((*Lr)[i].ptr,(*Lr)[i].lenght,&streams[st]);
	}

    checkCudaErrors(cudaStreamSynchronize(streams[st]));
#endif


#if defined DYNAMIC || ADAPTIVE
	if(IS_SAMPLING(SIGNAL->WORK_TYPE_ID))
		checkCudaErrors(cudaStreamAddCallback(streams[st],(cudaStreamCallback_t)&CallBackStartSampling,SIGNAL,0));
	(*pry)->addDeviceAssign(deviceId,1);
#endif

	genericWideKernel<<< cdp.multiProcessorCount * 5, umin(cdp.warpSize * GET_SIMD(SIGNAL->WORK_TYPE_ID), cdp.maxThreadsPerBlock), 0, streams[st]>>> (SIGNAL->Work);
	CudaTest("Launching parallel wide kernel");

#if defined DYNAMIC || ADAPTIVE
	if(IS_SAMPLING(SIGNAL->WORK_TYPE_ID)) {
		checkCudaErrors(cudaStreamAddCallback(streams[st],(cudaStreamCallback_t)&CallBackStopSampling,SIGNAL,0));
	}
#endif


#if defined GAMA_CACHE
	for(int i=0; i< Lw->size(); i++){
		cache->uncachePtr((*Lw)[i].ptr,(*Lw)[i].lenght,&streams[st],CACHE_READ_WRITE);
	}
	for(int i=0; i< Lr->size(); i++){
		cache->uncachePtr((*Lr)[i].ptr,(*Lr)[i].lenght,&streams[st]);
	}

	delete Lw;
	delete Lr;
#endif

#ifdef __SYNC
	checkCudaErrors(cudaEventRecord(syncPoint,streams[st]));
	checkCudaErrors(cudaEventSynchronize(syncPoint));
	checkCudaErrors(cudaEventDestroy(syncPoint));
#endif

	__sync_fetch_and_sub((volatile unsigned int*)SIGNAL->in_flight,1);

}
