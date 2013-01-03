/*
 * RuntimeScheduler.cpp
 *
 *  Created on: Apr 3, 2012
 *      Author: jbarbosa
 */

#include <gamalib/gamalib.h>

#include "RuntimeScheduler.h"

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

//static inline int log2(long x) {
//  long y;
//  asm ( "\tbsr %1, %0\n"
//      : "=r"(y)
//      : "r" (x)
//  );
//  return y;
//}

RuntimeScheduler::RuntimeScheduler() {



	_gmem = new MemorySystem();
	LowLevelMemAllocator::_memSys=(MemorySystem*)_gmem;

	pthread_barrier_init(&SyncBarrier, NULL, TOTAL_DEVICES+1);

#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
	_ROB = new FIFOLockFreeQueue<work*>();

#if defined DYNAMIC || ADAPTIVE
	_pry = NULL;
#else
	_pry = new Proxy();
#endif

	PerformanceModel* setPMs[TOTAL_DEVICES];

	for(unsigned dev=0; dev < TOTAL_DEVICES; dev++)
		setPMs[dev] = new PerformanceModel();

	unsigned IDsGPUs[TOTAL_DEVICES];

	unsigned int GPU_TYPE = 0;

	for(unsigned dev=0; dev < TOTAL_DEVICES; dev++) {

		switch (DEVICE_TYPES[dev]) {
		case CPU_X86 :
			SysDevices[dev] = new DeviceX86(dev,(Proxy**) &_pry,setPMs[dev],(MemorySystem*)_gmem,&SyncBarrier);
			break;
		case GPU_CUDA:
			SysDevices[dev] = new DeviceCuda(dev,GPU_TYPE++,(Proxy**) &_pry,setPMs[dev],(MemorySystem*)_gmem,&SyncBarrier);
			IDsGPUs[GPU_TYPE-1] = dev;
			break;
		default:
			SysDevices[dev] = NULL;
			break;
		}

	}

	Workqueue<work, DEVICE_QUEUE_SIZE, CPU_X86>* queuesDEVs[TOTAL_DEVICES];

	for(unsigned dev=0; dev < TOTAL_DEVICES; dev++)
		queuesDEVs[dev] = &(SysDevices[dev]->INBOX);

#if defined DYNAMIC || ADAPTIVE

	//It defines an Proxy (entity which assigns work from the ROB to the local queues)
	_pry = new Proxy(_ROB, queuesDEVs, setPMs, GPU_TYPE, IDsGPUs );
	((Proxy*) _pry)->launch();

	//Synchronism on Proxy's thread:
#ifndef __APPLE__
	while(_pry->cond1.__size[0] == 0){}
#else

#endif
#endif
#endif


#if defined(DEMAND_DRIVEN)
	ds = new DemandScheduler();

	unsigned int GPU_ID = 0;

	for(unsigned dev=0; dev < TOTAL_DEVICES; dev++) {

		switch (DEVICE_TYPES[dev]) {
		case CPU_X86 :
			SysDevices[dev] = new DeviceX86(dev,(MemorySystem*)_gmem,ds,&SyncBarrier);
			ds->setQueueDevice(dev,&SysDevices[dev]->INBOX);
			break;
		case GPU_CUDA:
			SysDevices[dev] = new DeviceCuda(dev,GPU_ID++,(MemorySystem*)_gmem,ds,&SyncBarrier);
			ds->setQueueDevice(dev,&SysDevices[dev]->INBOX);
			break;
		default:
			SysDevices[dev] = NULL;
			break;
		}


	}

	pthread_barrier_wait(&SyncBarrier);

#endif

	synchronize();

}

RuntimeScheduler::~RuntimeScheduler() {

	for(unsigned dev=0; dev < TOTAL_DEVICES; dev++) {
		SysDevices[dev]->sendSignal(TERMINATE_SIGNAL);
		SysDevices[dev]->join();
	}
#ifdef DEMAND_DRIVEN
	delete ds;
#endif
}


void RuntimeScheduler::synchronize() {

#ifdef DEMAND_DRIVEN
	for(unsigned dev=0; dev < TOTAL_DEVICES; dev++) {
		SysDevices[dev]->sendSignal(SYNC_SIGNAL);
	}

	pthread_barrier_wait(&SyncBarrier);
	X86MemFence();

#else
    for(unsigned dev=0; dev < TOTAL_DEVICES; dev++) {
            SysDevices[dev]->sendSignal(SYNC_SIGNAL);
    }

    for(unsigned dev=0; dev < TOTAL_DEVICES; dev++) {
            SysDevices[dev]->waitForStatus(DEVICE_KERNEL_SYNCED);
    }

    for(unsigned dev=0; dev < TOTAL_DEVICES; dev++) {
            SysDevices[dev]->resetSignal();
    }

#endif
}

bool RuntimeScheduler::submit(work* w) {

	X86MemFence();
#if defined(DEMAND_DRIVEN)
	if(TOTAL_DEVICES == 1) {
		ds->enqueue(w);
		return true;
	}
#endif

	//unsigned int part = umin(4096, ( 2l << roundUpToNextPowerOfTwo(TOTAL_DEVICES)));
	unsigned int part = roundUpToNextPowerOfTwo(TOTAL_DEVICES) << 4;
	List<work*>* l = (w->*DICE_CPU_TABLE[w->getWorkTypeID()])(part);

#if defined(DEMAND_DRIVEN)
	ds->enqueueMultiple(l);
	delete l;
	for(unsigned dev=0; dev < TOTAL_DEVICES; dev++) {
		SysDevices[dev]->signalWork();
	}
#endif

#if defined(DYNAMIC) || defined(ADAPTIVE)
	//Join with step bellow:
	while(!_ROB->trylock());

	unsigned int i = 0;
	unsigned int size = l->_size;
	unsigned int sizeSampling = size*0.2;


	for(;i<size ; i++)  {

		if(i<sizeSampling ) {
			SWAP_SAMPLING((*l)[i]->WORK_TYPE_ID);
		}
		_ROB->enqueue((*l)[i]);

	}

	flag++;

	((Proxy*) _pry)->setGLEmpty(false,w->getWorkTypeID());
	((Proxy*) _pry)->setDTasksROBWT(size,w->getWorkTypeID());

	_ROB->release();

	double end = getTimeMS();
	//    printf("Dice and ROB enqueue time: %.4f\n",(end-start));



#endif

#ifdef STATIC

	int loads[TOTAL_DEVICES];// = {52,19,19,19,19};

	for( unsigned int dev = 0 ; dev < TOTAL_DEVICES ; dev++ )
		loads[dev]= LOADS[dev]*_ROB->size();


	work* wu = NULL;

	for( unsigned int dev = 0 ; dev < TOTAL_DEVICES ; dev++ )

		for( int it = 0; it < loads[dev] && !_ROB->isEmpty() ; ){

			while(SysDevices[dev]->INBOX.isFull());

			if(_ROB->dequeue(wu)){
				SysDevices[dev]->INBOX.enqueue(wu);
				it++;
			}
		}

#endif

	return true;

}
