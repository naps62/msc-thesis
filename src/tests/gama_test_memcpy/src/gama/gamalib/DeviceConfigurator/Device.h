/*
 * DeviceConfig.h
 *
 *  Created on: Apr 3, 2012
 *      Author: jbarbosa
 */

#ifndef DEVICECONFIG_H_
#define DEVICECONFIG_H_

#include <config/workqueue.cfg.h>

#include <gamalib/utils/x86_utils.h>
#include <gamalib/memlib/smartpointer.h>
#include <gamalib/work/work.h>
#include <gamalib/workqueues/Workqueue.h>
#include <gamalib/workqueues/FifoLockFreeQueue.h>
#include <gamalib/Communication/channel.h>

#include <gamalib/Scheduling/DemandScheduler.h>
#include <gamalib/PerformanceModel/proxy.h>

class Device {
public:

	unsigned int deviceId;
	DEVICE_TYPE Type;
	unsigned int numberOfCores;
	volatile ComChannel channel;
	pthread_barrier_t *SyncBarrier;

#if defined(DEMAND_DRIVEN)
	DemandScheduler *ds;
	pthread_cond_t NoWork;
	pthread_mutex_t _NWLock;
	volatile unsigned int unlockSignal;
#endif

	Workqueue< work , DEVICE_QUEUE_SIZE, CPU_X86> INBOX;
	Workqueue< work , DEVICE_QUEUE_SIZE, CPU_X86> OUTBOX;

#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
	Proxy** pry;
	PerformanceModel* pm;
#endif
	MemorySystem* _memSys;

public:
#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
	Device(unsigned int device, DEVICE_TYPE T, Proxy** _pry, PerformanceModel* _pm, MemorySystem* memSys,pthread_barrier_t *SB) : deviceId(device), Type(T), _memSys(memSys), SyncBarrier(SB) {
		pry = (Proxy**) _pry;
		pm = _pm;
	};
#endif


#if defined(DEMAND_DRIVEN)
	Device(unsigned int device, DEVICE_TYPE T, MemorySystem* memSys, DemandScheduler *s, pthread_barrier_t *SB) : deviceId(device), Type(T), _memSys(memSys), ds(s), SyncBarrier(SB) {
		pthread_mutex_init(&_NWLock,NULL);
		pthread_cond_init(&NoWork,NULL);
		unlockSignal=0;
	};
#endif

	virtual ~Device() {};

	__forceinline void sendSignal(SIGNAL_DEVICE _signal) {
#ifdef DEMAND_DRIVEN
		pthread_cond_signal(&NoWork);
#endif
		channel.toDevice = _signal;
	}

#ifdef DEMAND_DRIVEN
	__forceinline void signalWork() {
		unlockSignal=1;
		pthread_cond_signal(&NoWork);
	}
#endif

	__forceinline void sendStatus(DEVICE_STATUS _status) {
		channel.fromDevice = _status;
	}

	__forceinline void waitForStatus(DEVICE_STATUS _wait_status) {
		while(channel.fromDevice != _wait_status);
	}

	__forceinline void sendSignalAndWait(SIGNAL_DEVICE _signal, DEVICE_STATUS _wait_status) {
		channel.toDevice = _signal;
		waitForStatus(_wait_status);
	}

	__forceinline void resetSignal() {
		channel.toDevice = NULL_SIGNAL;
	}

	__forceinline void resetStatus() {
		channel.fromDevice = DEVICE_NULL_STATUS;
	}

	__forceinline void sendWaitSync() {
		channel.fromDevice = DEVICE_KERNEL_SYNCED;
		while(channel.toDevice != NULL_SIGNAL);
		resetStatus();
	}

	void* run() {
		return NULL;
	}
	static void* threadproc(void*) {
		printf("Called wrong");
		return NULL;
	}

#ifdef DEMAND_DRIVEN
	__forceinline void waitForWork() {
		pthread_mutex_lock(&_NWLock);
		struct timespec   ts;
		struct timeval    tp;
	    gettimeofday(&tp, NULL);
	    ts.tv_sec  = tp.tv_sec;
	    ts.tv_nsec = tp.tv_usec * 1000 + 100;
		while(unlockSignal==0) pthread_cond_timedwait(&NoWork,&_NWLock,&ts);
		pthread_mutex_unlock(&_NWLock);
	}
#endif

	const __inline__ pthread_t&  getTID() const { return m_tid; }

	bool __inline__ join()
	{
		int nerr = ::pthread_join(getTID(), 0);
		return (nerr == 0);
	}
protected:

	Device(const Device&);                // no implementation
	Device& operator=(const Device&);    // no implementation

	pthread_t m_tid;
	pthread_attr_t attr;

};

#endif /* DEVICECONFIG_H_ */
