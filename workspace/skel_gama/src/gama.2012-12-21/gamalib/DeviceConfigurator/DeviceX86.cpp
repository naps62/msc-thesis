/*
 * DeviceX86Conf.cpp
 *
 *  Created on: Apr 3, 2012
 *      Author: jbarbosa
 */
#include <config/common.h>

#include <pthread.h>
#include <stdio.h>
#include <omp.h>
#include <config/vtable.h>

#include <gamalib/utils/x86_utils.h>
#include "Device.h"
#include "DeviceX86.h"

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

__thread unsigned int DeviceID_CPU;
void* _memCPU;

#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
DeviceX86::DeviceX86(unsigned int deviceID, Proxy** pry, PerformanceModel* pm, MemorySystem* memSys,pthread_barrier_t *SB) : Device(deviceID,GPU_CUDA,pry,pm, memSys,SB)
#endif
#if defined(DEMAND_DRIVEN)
DeviceX86::DeviceX86(unsigned int deviceID, MemorySystem* memSys, DemandScheduler* s,pthread_barrier_t *SB) : Device(deviceID,CPU_X86,memSys,s,SB)
#endif
{
	::pthread_attr_init(&attr);
	::pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	::pthread_create(&m_tid, NULL,  DeviceX86::threadproc, (void*) this);

}

DeviceX86::~DeviceX86() {
}
#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
void* DeviceX86::run(void) {

	DeviceID_CPU = deviceId;
	double start, end;

	_memCPU = _memSys;
	float executionTime;

	//	pthread_barrier_wait(SyncBarrier);

	while(channel.toDevice != TERMINATE_SIGNAL || ! INBOX.isEmpty() ) {

		if(!INBOX.isEmpty()) {
			work* w_item = NULL;
			if(INBOX.dequeue(w_item) && w_item != NULL) {
#if defined DYNAMIC || ADAPTIVE
				if(IS_SAMPLING(w_item->WORK_TYPE_ID))
					start = getTimeMS();
#endif

				(w_item->*WORK_CPU_TABLE[w_item->getWorkTypeID()])();

#if defined DYNAMIC || ADAPTIVE
				if(IS_SAMPLING(w_item->WORK_TYPE_ID)){

					end = getTimeMS();
					executionTime = end-start;
					pm->putTaskAndTime(w_item->getWorkTypeID(),128,executionTime);
				}

				(*pry)->addExecution(deviceId,w_item->getWorkTypeID());
#endif
				LowLevelMemAllocator::dealloc(w_item,SHARED);
			}
		}

#if defined DYNAMIC || ADAPTIVE

		if( (*pry) )
			if( (INBOX.getSize()<SCHEDULER_MINIMUM) ) {
				(*pry)->assign();
			}

		if( (*pry) )
			if( INBOX.isEmpty() && channel.toDevice == SYNC_SIGNAL && (*pry)->isGlEmpty() ) {
				sendWaitSync();
			}

#else

		if(INBOX.isEmpty() && channel.toDevice == SYNC_SIGNAL ){
			sendWaitSync();
		}

#endif

	}

#if defined DYNAMIC || ADAPTIVE
	printf("History of device %d (CPU_X86):\n",deviceId);
	(*pry)->printHistory(deviceId);
	printf("\n");
#endif

	pthread_exit(NULL);
}
#endif

#if defined(DEMAND_DRIVEN)
void* DeviceX86::run(void) {

	DeviceID_CPU = deviceId;
	double start, end;

	_memCPU = _memSys;
	float executionTime;

	channel.fromDevice = DEVICE_KERNEL_RUNNING;
	channel.toDevice = NULL_SIGNAL;
	pthread_barrier_wait(SyncBarrier);


	while(channel.toDevice != TERMINATE_SIGNAL || ! INBOX.isEmpty() || !ds->isEmpty() ) {

		if(INBOX.getSize() < SCHEDULER_CHUNK) {
			std::vector<work*>* aw = ds->request(deviceId,1.f);
			for(unsigned ii=0; ii < aw->size(); ii++) {
				work* w = aw->at(ii);
				INBOX.enqueue(w);
			}
			delete aw;

		}

		if(!INBOX.isEmpty()) {
			work* w_item = NULL;
			if(INBOX.dequeue(w_item) && w_item != NULL) {
				(w_item->*WORK_CPU_TABLE[w_item->getWorkTypeID()])();
				LowLevelMemAllocator::dealloc(w_item,SHARED);
			}
		}

		if(INBOX.isEmpty() && ds->isEmpty() && channel.toDevice == SYNC_SIGNAL ){
			channel.toDevice = NULL_SIGNAL;
			pthread_barrier_wait(SyncBarrier);
			channel.fromDevice = DEVICE_KERNEL_RUNNING;
		}
	}

	pthread_exit(NULL);
}
#endif

void* DeviceX86::threadproc(void* a_param) {
	DeviceX86* pthread = static_cast<DeviceX86*>(a_param);
	return pthread->run();
}
