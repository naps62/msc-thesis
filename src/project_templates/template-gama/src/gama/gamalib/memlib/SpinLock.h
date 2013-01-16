/*
 * SpinLock.h
 *
 *  Created on: May 12, 2011
 *      Author: jbarbosa
 */

#ifndef SPINLOCK_H_
#define SPINLOCK_H_

#include <helper_cuda.h>
#include <helper_math.h>
#include <config/common.h>
#include<gamalib/utils/x86_utils.h>

struct SpinLock {

#ifndef __CUDACC__
	volatile
#endif
	int mutex;

	__DEVICE__  SpinLock() { mutex=-1; };

	__DEVICE__ __forceinline ~SpinLock() {};

	__DEVICE__ __forceinline bool Try(int id) {
		return __any(atomicCAS(&mutex, -1, id)==-1);
	}

	__DEVICE__ __forceinline bool Acquire(int id) {
		while(!Try(id));
		return true;
	}

	__DEVICE__ __forceinline bool Release(int id) {
		atomicCAS(&mutex, id, -1);
		return true;

	}
};

#endif /* SPINLOCK_H_ */
