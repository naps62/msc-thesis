/*
 * DeviceX86Conf.h
 *
 *  Created on: Apr 3, 2012
 *      Author: jbarbosa
 */

#ifndef DEVICEX86CONF_H_
#define DEVICEX86CONF_H_


#include <config/workqueue.cfg.h>

#include "Device.h"

class DeviceX86: public Device {
public:


#if defined(STATIC) || defined(DYNAMIC) || defined(ADAPTIVE)
	DeviceX86(unsigned int,Proxy**,PerformanceModel* pm, MemorySystem* memSys,pthread_barrier_t *SB);
#endif

#if defined(DEMAND_DRIVEN)
	DeviceX86(unsigned int, MemorySystem*, DemandScheduler*, pthread_barrier_t*);
#endif
	virtual ~DeviceX86();


	void* run();
	static void* threadproc(void*);
	void* initExec();

};

#endif /* DEVICEX86CONF_H_ */
