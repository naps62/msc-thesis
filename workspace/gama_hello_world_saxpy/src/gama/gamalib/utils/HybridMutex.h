/*
 * HybridMutex.h
 *
 *  Created on: Jun 22, 2012
 *      Author: ricardo
 */

#ifndef HYBRIDMUTEX_H_
#define HYBRIDMUTEX_H_

#include <gamalib/memlib/SpinLock.h>
#include <gamalib/utils/GlobalMutex.h>

template <unsigned int DEVICES> class HybridMutex{

public:
	SpinLock slock;
	GlobalMutex<DEVICES> glock;

	__DEVICE__ HybridMutex(){
	}

	__DEVICE__ ~HybridMutex(){
	}


	__DEVICE__
	bool Acquire(unsigned int id) {
		bool status = slock.Acquire(id);
		if(!status) return false;
		status = glock.Acquire(DeviceID);
		if(!status){ slock.Release(id); return false;}
		return true;
	}

	__DEVICE__
	bool Release(unsigned int id) {
		slock.Release(id);
		glock.Release(DeviceID);
		return true;
	}
};


#endif /* HYBRIDMUTEX_H_ */
