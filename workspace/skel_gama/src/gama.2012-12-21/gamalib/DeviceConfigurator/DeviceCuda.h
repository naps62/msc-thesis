/*
 * DeviceCudaConf.h
 *
 *  Created on: Apr 2, 2012
 *      Author: jbarbosa
 */


#ifndef DEVICECUDACONF_H_
#define DEVICECUDACONF_H_

#include <cuda.h>
#include <helper_cuda.h>

#include <config/workqueue.cfg.h>

#include "Device.h"

//#include <gamalib/memlib/DeviceCacheCuda.h>
//#include <gamalib/memlib/Cache.h>
#include <gamalib/cache/cacheControl.h>


struct Information {


	unsigned long WORK_TYPE_ID;
	work*		  Work;
	Workqueue<  work ,INBOX_QUEUE_SIZE, CPU_X86>* INBOX;
	unsigned int long deviceID;
	PerformanceModel* pm;
	Proxy** pry;

	cacheControl<GPU_CUDA>* cache;

	volatile int* in_flight;

	double _start, _end;

	Information(unsigned long _WORKID, work* _work, Workqueue<  work ,INBOX_QUEUE_SIZE, CPU_X86>* _INBOX, PerformanceModel* _pm, Proxy** _pry, unsigned int devID, cacheControl<GPU_CUDA>* c) :
		WORK_TYPE_ID(_WORKID), Work(_work), INBOX(_INBOX), pm(_pm), pry(_pry), deviceID(devID), cache(c) {

	}
	__forceinline
	void start() {
		_start = getTimeMS();
	}
	__forceinline
	void stop() {
		_end = getTimeMS();
	}
	__forceinline
	double elapse() {
		return  _end-_start;//;_end.tv_sec*1000.0 + _end.tv_usec/1000.0 - _start.tv_sec*1000.0 - _start.tv_usec/1000.0;
	}

};


class DeviceCuda : public Device {

public:

	unsigned int cudaDeviceID;
	CUcontext cuContext;
	CUdevice cuDevice;
	cudaStream_t streams[32];
	volatile int kernels_in_flight;
	int kernels_off_flight;

	cacheControl<GPU_CUDA>* cache;

	cudaDeviceProp cdp;
	unsigned int block;
	volatile unsigned int stream;
public:
#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
	DeviceCuda(unsigned int,unsigned int,Proxy**,PerformanceModel* pm, MemorySystem* memSys,pthread_barrier_t *SB);
#endif

#if defined(DEMAND_DRIVEN)
	DeviceCuda(unsigned int,unsigned int, MemorySystem*, DemandScheduler*, pthread_barrier_t*);
#endif
	virtual ~DeviceCuda();

	void* run();

	static void* threadproc(void*);

	static unsigned int cudaDevices() {
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		return deviceCount;
	}

	void classWideKernel(Information* SIGNAL);
	static void* threadWideKernel(void*);
	void* initExec();

};


struct ThreadParam {
	DeviceCuda* devicePtr;
	Information* SIGNAL;
};
#endif /* DEVICECUDACONF_H_ */
