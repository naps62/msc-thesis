/*
 * Workqueue.hu
 *
 *  Created on: Apr 6, 2012
 *      Author: jbarbosa
 */

#ifndef WORKQUEUE_HU_
#define WORKQUEUE_HU_

#include <config/common.h>
#include <gamalib/memlib/LowLevelMemAllocator.h>
#include <gamalib/memlib/smartpointer.h>
#include "Workqueue.h"


template <typename T, unsigned int LENGTH>
class Workqueue<T,LENGTH,GPU_CUDA> {
public:
	volatile MEM_TYPE MT;
	T* data[LENGTH];
	volatile unsigned int _enqueue;
	volatile unsigned int _dequeue;


public:

	__DEVICE__ Workqueue() : _enqueue(0), _dequeue(0) {

	}

	__DEVICE__ Workqueue(MEM_TYPE _MT) : MT(_MT), _enqueue(0), _dequeue(0) {

	}

	__DEVICE__ virtual ~Workqueue() {};

	__DEVICE__ bool dequeue(T *&elem) {
		elem = NULL;

		if(_enqueue > _dequeue) {
			unsigned int pos = atomicAdd((int*)&_dequeue,1);
			elem = data[pos];
			return true;
		}

		return false;
	}
	__DEVICE__ bool enqueue(T *elem) {
		unsigned int size = ((_enqueue-_dequeue) & (LENGTH-1));
		if(size < LENGTH) {
			unsigned int pos = atomicAdd((int*)&_enqueue,1);
			data[pos] = elem;
			return true;
		}
		return false;
	}

	__DEVICE__ unsigned getSize() { return (_enqueue - _dequeue); }
	__DEVICE__ bool isEmpty() { printf("Here..."); return (_enqueue ==_dequeue); }
	__DEVICE__ bool isFull() { return (_enqueue-_dequeue) >= LENGTH; }

	__DEVICE__
	void *operator new(size_t s, MEM_TYPE MT) {
		assert(MT == DEVICE);
		Workqueue<T,LENGTH,GPU_CUDA>* ptr = malloc(s);
		return ptr;
	}

	__DEVICE__
	void *operator new[](size_t s, MEM_TYPE MT) {
		assert(MT == DEVICE);
		Workqueue<T,LENGTH,GPU_CUDA>* ptr =  malloc(s);
		return ptr;
	}
	__DEVICE__
	void operator delete[](void *dp) {
		assert(((Workqueue<T,LENGTH,GPU_CUDA>*)dp)->MT == DEVICE);
		return free(dp);

	}

	__DEVICE__
	void operator delete(void *dp) {
		assert(((Workqueue<T,LENGTH,GPU_CUDA>*)dp)->MT == DEVICE);
		return free(dp);
	}

};
//
//template <typename T, unsigned int LENGTH>
//__DEVICE__ bool Workqueue<T,LENGTH,GPU_CUDA>::dequeue(T *elem) {
//	elem=NULL;
//	if(_enqueue > _dequeue) {
//		__shared__ unsigned int pos;
//		if(threadIdx.x & 31 == 0) pos = atomicAdd((int*)&_dequeue,1);
//		elem = data[pos+(threadIdx.x & 31)];
//		return true;
//	}
//	return false;
//}
//
//
//template <typename T, unsigned int LENGTH>
//__DEVICE__ bool Workqueue<T,LENGTH,GPU_CUDA>::enqueue(T *elem) {
//	unsigned int size = ((_enqueue-_dequeue) & (LENGTH-1));
//	if(size < LENGTH) {
//		__shared__ unsigned int pos;
//		if(threadIdx.x & 31 == 0) pos = atomicAdd((int*)&_enqueue,1);
//		data[pos+(threadIdx.x & 31)] = &elem;
//		return true;
//	}
//	return false;
//}



#endif
