/*
 * DeviceCudaConf.cpp
 *
 *  Created on: Apr 2, 2012
 *      Author: jbarbosa
 */

#include <vector>

#include <config/common.h>
#include <pthread.h>
#include <stdio.h>

#include <helper_cuda.h>

#include <gamalib/utils/x86_utils.h>

#include <config/vtable.h>
#include "DeviceCuda.h"

__inline__ unsigned int roundUpToNextPowerOfTwo(unsigned int x)
{
	x--;
	x |= x >> 1;  // handle  2 bit numbers
	x |= x >> 2;  // handle  4 bit numbers
	x |= x >> 4;  // handle  8 bit numbers
	x |= x >> 8;  // handle 16 bit numbers
	x |= x >> 16; // handle 32 bit numbers
	x++;

	return x;
}
extern pthread_barrier_t barrScheduler;

extern void initCudaDevice(unsigned int DeviceID, MemorySystem* mem);

#define PTHREAD_STREAM


#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
DeviceCuda::DeviceCuda(unsigned int deviceID, unsigned int _cudaDeviceID, Proxy** pry, PerformanceModel* pm, MemorySystem* memSys,pthread_barrier_t *SB) : Device(deviceID,GPU_CUDA,pry,pm, memSys,SB), cudaDeviceID(_cudaDeviceID), kernels_in_flight(0),stream(0)
#endif
#if defined(DEMAND_DRIVEN)
DeviceCuda::DeviceCuda(unsigned int deviceID,unsigned int _cudaDeviceID, MemorySystem* memSys, DemandScheduler* s, pthread_barrier_t *SB) : Device(deviceID,GPU_CUDA,memSys,s,SB), cudaDeviceID(_cudaDeviceID), kernels_in_flight(0), stream(0)
#endif
{
	int deviceCount = DeviceCuda::cudaDevices();
	if (deviceCount == 0 || (unsigned)deviceCount <= cudaDeviceID) {
		printf("There is no %u device supporting CUDA.\n", cudaDeviceID);
		exit (0);
	}

	checkCudaErrors(cudaSetDevice(cudaDeviceID));
	checkCudaErrors(cudaGetDeviceProperties(&cdp,cudaDeviceID));
	numberOfCores = cdp.multiProcessorCount;
	unsigned wide = _ConvertSMVer2Cores(cdp.major, cdp.minor);


	block = roundUpToNextPowerOfTwo(14);

	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	printf("Device %s CORES: %u WS: %u BK: %u\n", cdp.name, numberOfCores, cdp.warpSize, block);

	for(int st=0; st < 32; st++) checkCudaErrors(cudaStreamCreate(&(streams[st])));

#if defined GAMA_CACHE
	kernels_in_flight=0;
#endif
	initCudaDevice(deviceID, LowLevelMemAllocator::_memSys);


#if defined GAMA_CACHE
	cache = new cacheControl<GPU_CUDA>();
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	::pthread_attr_init(&attr);
	::pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	::pthread_create(&m_tid, &attr,  DeviceCuda::threadproc, (void*) this);
}

#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
void* DeviceCuda::run(void) {

	cudaSetDevice(cudaDeviceID);
	struct timeval start, end;
	float executionTime;

	unsigned int CK = 32;
	kernels_in_flight = 0;
	while(channel.toDevice != TERMINATE_SIGNAL || ! INBOX.isEmpty()) {

		if(kernels_in_flight < CK && !INBOX.isEmpty()) {
			work* w_item = NULL;
			if(INBOX.dequeue(w_item) && w_item != NULL) {

					Information* inf = new Information(w_item->WORK_TYPE_ID, w_item, NULL, pm, pry, deviceId, cache);
					inf->in_flight=&kernels_in_flight;
					inf->start();
					pthread_t w_tid;
					pthread_attr_t w_attr;

					ThreadParam* p = new ThreadParam();
					p->devicePtr = this;
					p->SIGNAL = inf;
					__sync_fetch_and_add((volatile unsigned int*)&kernels_in_flight,1);
					::pthread_attr_init(&w_attr);
					::pthread_attr_setdetachstate(&w_attr, PTHREAD_CREATE_DETACHED);
					::pthread_create(&w_tid, &w_attr,  DeviceCuda::threadWideKernel, (void*) p);

#if defined DYNAMIC || ADAPTIVE
				(*pry)->addExecution(deviceId,w_item->getWorkTypeID());
#endif

			}
		}

#if defined DYNAMIC || ADAPTIVE

		if( (*pry) )
			if( ((INBOX.getSize()+(*pry)->queryDriverQueueSize(deviceId))<SCHEDULER_MINIMUM) && ((volatile int)(kernels_in_flight))==0 ) {
				(*pry)->assign();
			}

		if( (*pry) )
			if(INBOX.isEmpty() && channel.toDevice == SYNC_SIGNAL && (*pry)->isGlEmpty() && ((volatile int)(kernels_in_flight))==0){
				checkCudaErrors(cudaDeviceSynchronize());
				cache->cacheReset();
				sendWaitSync();
			}

#else

		if(INBOX.isEmpty() && channel.toDevice == SYNC_SIGNAL && kernels_in_flight==0){
			for(int i=0; i < 32; i++) {
				checkCudaErrors(cudaStreamSynchronize(streams[i]));
			}
			cache->cacheReset();
			sendWaitSync();

		}

#endif

	}

	checkCudaErrors(cudaDeviceSynchronize());


#if defined DYNAMIC || ADAPTIVE
	printf("History of device %d (GPU_CUDA):\n",deviceId);
	(*pry)->printHistory(deviceId);
	printf("\n");
#endif

	pthread_exit(NULL);
}
#endif

#if defined(DEMAND_DRIVEN)

void* DeviceCuda::run(void) {

	cudaSetDevice(cudaDeviceID);

	struct timeval start, end;
	float executionTime;

	unsigned int CK = 8;

	printf("CK: %u\n",CK);

	channel.fromDevice = DEVICE_KERNEL_RUNNING;
	channel.toDevice = NULL_SIGNAL;

	pthread_barrier_wait(SyncBarrier);

	bool request = true;

	while(channel.toDevice != TERMINATE_SIGNAL || ! INBOX.isEmpty() || !ds->isEmpty()) {

		if(INBOX.getSize() < SCHEDULER_CHUNK) {
			std::vector<work*>* aw = ds->request(deviceId,1.f);
			if(aw->size() !=0) {
				for(unsigned ii=0; ii < aw->size(); ii++) {
					INBOX.enqueue(aw->at(ii));
				}
				delete aw;
			}
		}

		if( kernels_in_flight < CK && !INBOX.isEmpty()) {
			work* w_item = NULL;
			if(INBOX.dequeue(w_item) && w_item != NULL) {
					Information* inf = new Information(w_item->WORK_TYPE_ID, w_item, NULL, NULL, NULL, deviceId, cache);
					inf->in_flight=&kernels_in_flight;
					inf->start();
					pthread_t w_tid;
					pthread_attr_t w_attr;

					__sync_fetch_and_add((volatile unsigned int*)&kernels_in_flight,1);

					ThreadParam* p = new ThreadParam();
					p->devicePtr = this;
					p->SIGNAL = inf;

					::pthread_attr_init(&w_attr);
					::pthread_attr_setdetachstate(&w_attr, PTHREAD_CREATE_DETACHED);
					::pthread_create(&w_tid, &w_attr,  DeviceCuda::threadWideKernel, (void*) p);
			}
		}

		while(kernels_in_flight == CK);

		if(INBOX.isEmpty() && ds->isEmpty() && channel.toDevice == SYNC_SIGNAL && ((volatile int)(kernels_in_flight))==0){
			channel.toDevice = NULL_SIGNAL;
			checkCudaErrors(cudaDeviceSynchronize());
#if defined GAMA_CACHE
			cache->cacheReset();
#endif
			pthread_barrier_wait(SyncBarrier);
			channel.fromDevice = DEVICE_KERNEL_RUNNING;
		}



	}

	checkCudaErrors(cudaDeviceSynchronize());
	pthread_exit(NULL);

}
#endif

void* DeviceCuda::threadproc(void* a_param) {
	DeviceCuda* pthread = static_cast<DeviceCuda*>(a_param);
	return pthread->run();
}

void* DeviceCuda::threadWideKernel(void* a_param) {
	ThreadParam* pthread = static_cast<ThreadParam*>(a_param);

	pthread->devicePtr->classWideKernel(pthread->SIGNAL);
	//LowLevelMemAllocator::dealloc(pthread->SIGNAL->Work,SHARED);

	delete pthread->SIGNAL;
	delete pthread;
	return NULL;
}

DeviceCuda::~DeviceCuda() {
	for(int st=0; st < 32; st++) checkCudaErrors(cudaStreamDestroy(streams[st]));
	//cutilDrvSafeCall(cuCtxDestroy(cuContext));
}


void CallBackStartSampling(CUstream st, cudaError_t error, void* userData) {
	Information* inf = (Information*)userData;
	inf->start();
}

void CallBackStopSampling(CUstream st, cudaError_t error, void* userData) {
	Information* inf = (Information*)userData;
	inf->stop();
}


void CallBackGeneric(CUstream st, cudaError_t error, void* userData) {
	Information* inf = (Information*)userData;
#if defined DYNAMIC || ADAPTIVE
	if(IS_SAMPLING(inf->Work->WORK_TYPE_ID)){
		inf->pm->putTaskAndTime(inf->Work->getWorkTypeID(),128,inf->elapse());
	}
	(*(inf->pry))->addDeviceExecution(inf->deviceID);
#endif

	delete inf;
}


void CallBackWide(CUstream st, cudaError_t error, void* userData) {
	Information* inf = (Information*)userData;
#if defined DYNAMIC || ADAPTIVE


	if(IS_SAMPLING(inf->Work->WORK_TYPE_ID)){
		inf->pm->putTaskAndTime(inf->Work->getWorkTypeID(),128,inf->elapse());
	}
	(*(inf->pry))->addDeviceExecution(inf->deviceID);
#endif

	LowLevelMemAllocator::dealloc(inf->Work,SHARED);
}
