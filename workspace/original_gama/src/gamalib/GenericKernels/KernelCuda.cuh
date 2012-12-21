/*
 * KernelCuda.cu
 *
 *  Created on: Apr 6, 2012
 *      Author: jbarbosa
 */
#include <config/common.h>
#include <gamalib/gamalib.h>
#include <gamalib/gamalib.cuh>


const float G = 6.67e-11;



extern "C"
__global__ void initDevice() {
	__syncthreads();
}



extern "C"
__global__ void saxpy(smartPtr<float> X,smartPtr<float> Y,smartPtr<float> R, float alpha, unsigned long N) {
	int index = blockIdx.x * SPLIT + threadIdx.x;

	for(int i = index; (i < index+SPLIT && i < N) ; i+= NTHREAD) {
		R[i] = X[i] * alpha + Y[i];
	}
}

void initCudaDevice(unsigned int deviceId, MemorySystem* mem) {
	initDevice<<<1,1>>>();
	checkCudaErrors(cudaMemcpyToSymbol(DeviceID_GPU,&deviceId,sizeof(unsigned int),0,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(_memGPU,&mem,sizeof(MemorySystem*),0,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
}

void callSaxpy(unsigned NBLOCKS, unsigned NTHREADS,smartPtr<float> X,smartPtr<float> Y,smartPtr<float> R, float alpha, unsigned long N) {
	saxpy<<<ceil(N/SPLIT),NTHREADS>>>(X,Y,R,alpha,N);
	//cutilCheckMsg("Running kernel kernel");
	checkCudaErrors(cudaDeviceSynchronize());

}

