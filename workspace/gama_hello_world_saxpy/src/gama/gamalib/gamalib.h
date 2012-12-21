/*
 * gamalib.h
 *
 *  Created on: Apr 2, 2012
 *      Author: jbarbosa
 */

#ifndef GAMALIB_H_
#define GAMALIB_H_

#include <deque>

#include <config/common.h>

#include <gamalib/memlib/LowLevelMemAllocator.h>
#include <gamalib/memlib/MemorySystem.h>
#include <gamalib/memlib/smartpointer.h>

#include <gamalib/cache/cache.h>

#include <gamalib/DeviceConfigurator/Device.h>
#include <gamalib/DeviceConfigurator/DeviceX86.h>
#include <gamalib/DeviceConfigurator/DeviceCuda.h>



#include <gamalib/utils/x86_utils.h>

#include <gamalib/workqueues/Workqueue.h>
#include <gamalib/workqueues/FifoLockFreeQueue.h>
#include <gamalib/Datastructures/List.h>
#include <gamalib/Scheduling/RuntimeScheduler.h>
#include <gamalib/work/work.h>

#include <config/vtable.h>

#endif /* GAMALIB_H_ */
