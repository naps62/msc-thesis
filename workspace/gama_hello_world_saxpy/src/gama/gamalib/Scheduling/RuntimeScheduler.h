/*
 * RuntimeScheduler.h
 *
 *  Created on: Apr 3, 2012
 *      Author: jbarbosa
 */

#ifndef RUNTIMESCHEDULER_H_
#define RUNTIMESCHEDULER_H_

#include <config/common.h>

#include <gamalib/Scheduling/DemandScheduler.h>

#include <gamalib/PerformanceModel/proxy.h>

using namespace std;

class RuntimeScheduler {

	Device* SysDevices[TOTAL_DEVICES];

	pthread_barrier_t SyncBarrier;
	pthread_cond_t NoWork;

#if defined(DEMAND_DRIVEN)
	DemandScheduler *ds;
#endif

#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
	FIFOLockFreeQueue<work*>* _ROB;
	volatile Proxy* _pry;
	int flag;
#endif

public:
	RuntimeScheduler();
	virtual ~RuntimeScheduler();

	bool submit(work* w);

	void synchronize();

};

#endif /* RUNTIMESCHEDULER_H_ */
