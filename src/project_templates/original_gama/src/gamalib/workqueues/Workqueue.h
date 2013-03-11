/*
 * Workqueue.h
 *
 *  Created on: Apr 6, 2012
 *      Author: jbarbosa
 */

#ifndef WORKQUEUE_H_
#define WORKQUEUE_H_

template <typename T, unsigned int LENGTH, DEVICE_TYPE DT>
class Workqueue {

public:
	MEM_TYPE MT;
	T* data[LENGTH];
	unsigned int _enqueue;
	unsigned int _dequeue;

	SpinLock lock;


public:

	__DEVICE__ Workqueue() : _enqueue(0), _dequeue(0) {

	}
	__DEVICE__ Workqueue(MEM_TYPE _MT) : MT(_MT), _enqueue(0), _dequeue(0) {

	}
	__DEVICE__ virtual ~Workqueue() {};

	__DEVICE__ bool dequeue(T *&elem) {
		lock.Acquire(1);
		if(_enqueue > _dequeue) {
			unsigned int pos = atomicAdd((int*)&_dequeue,1);
			elem = data[pos%LENGTH];
			lock.Release(1);
			return true;
		}
		lock.Release(1);
		return false;
	}

	__DEVICE__ bool dequeueBack(T *&elem) {
		lock.Acquire(1);
		if(_enqueue > _dequeue) {
			elem = data[_enqueue%LENGTH];
			atomicDec((unsigned int*)&_enqueue,1);
			lock.Release(1);
			return true;
		}
		lock.Release(1);
		return false;
	}

//	__DEVICE__ T* dequeue() {
//
//		if(_enqueue > _dequeue) {
//			unsigned int pos = atomicAdd((int*)&_dequeue,1);
//			return data[pos];
//			//return elem;
//		}
//
//		return NULL;
//	}


//	__DEVICE__
//	bool dequeueMany(T *&elem, unsigned long num) {
//		if(_enqueue > _dequeue) {
//			unsigned int pos = atomicAdd((int*)&_dequeue,num);
//			elem = data[pos%LENGTH];
//			return true;
//		}
//		return false;
//	}


	__DEVICE__ bool enqueue(T *elem) {

		lock.Acquire(1);
		unsigned int size = ((_enqueue-_dequeue) & (LENGTH-1));

		if(size < LENGTH) {
			unsigned int pos = atomicAdd((int*)&_enqueue,1);
			data[pos%LENGTH] = elem;
			lock.Release(1);
			return true;
		}
		lock.Release(1);
		return false;
	}


	__DEVICE__ unsigned getSize() { return (_enqueue - _dequeue); }
	__DEVICE__ bool isEmpty() { printf("Here...\n"); return (_enqueue ==_dequeue); }
	__DEVICE__ bool isFull() { return (_enqueue - _dequeue) >= LENGTH; }


	__DEVICE__
	void *operator new(size_t s, MEM_TYPE MT) {return NULL;};

	__DEVICE__
	void *operator new[](size_t s, MEM_TYPE MT) {return NULL;};


	__DEVICE__
	void operator delete[](void *dp) {};

	__DEVICE__
	void operator delete(void *dp) {};
};

#ifndef __CUDACC__
template <typename T, unsigned int LENGTH>
class Workqueue<T,LENGTH,CPU_X86>{

public:
	MEM_TYPE MT;
	T* data[LENGTH];
	volatile unsigned int _enqueue;
	volatile unsigned int _dequeue;

	SpinLock lock;

public:

	Workqueue() : _enqueue(0), _dequeue(0) {
	}

	Workqueue(MEM_TYPE _MT) : MT(_MT), _enqueue(0), _dequeue(0) {
	}

	virtual ~Workqueue() {};

	bool dequeue(T *&elem) {
		lock.Acquire(1);
		if(_enqueue > _dequeue) {
			unsigned int pos = atomicAdd((int*)&_dequeue,1);
			elem = data[pos%LENGTH];
			lock.Release(1);
			return true;
		}
		lock.Release(1);
		return false;
	}

	T* dequeue() {
		lock.Acquire(1);
		if(_enqueue > _dequeue) {
			unsigned int pos = atomicAdd((int*)&_dequeue,1);
			lock.Release(1);
			return data[pos];
		}
		lock.Release(1);
		return NULL;
	}

	__DEVICE__ bool dequeueBack(T *&elem) {
		lock.Acquire(1);
		if(_enqueue > _dequeue) {
			elem = data[_enqueue%LENGTH];
			atomicDec((unsigned int*)&_enqueue,1);
			lock.Release(1);
			return true;
		}
		lock.Release(1);
		return false;
	}

	bool dequeueMany(T *&elem, unsigned long num) {
		lock.Acquire(1);
		if(_enqueue > _dequeue) {
			unsigned int pos = atomicAdd((int*)&_dequeue,num);
			elem = data[pos%LENGTH];
			lock.Release(1);
			return true;
		}
		lock.Release(1);
		return false;
	}

	bool enqueue(T *elem) {
		lock.Acquire(1);
		unsigned int size = ((_enqueue-_dequeue) & (LENGTH-1));

		if(size < LENGTH) {
			unsigned int pos = atomicAdd((int*)&_enqueue,1);
			data[pos%LENGTH] = elem;
			lock.Release(1);
			return true;
		}
		lock.Release(1);
		return false;
	}

	unsigned getSize() { return (_enqueue - _dequeue); }

	bool isEmpty() { return (_enqueue ==_dequeue); }
	bool isFull() { return (_enqueue-_dequeue) >= LENGTH; }


	void *operator new(size_t s, MEM_TYPE MT) {
		Workqueue<T,LENGTH,CPU_X86>* ptr = (Workqueue<T,LENGTH,CPU_X86>*)LowLevelMemAllocator::alloc(s,MT);
		 return ptr;
	}


	void *operator new[](size_t s, MEM_TYPE MT) {
		 Workqueue<T,LENGTH,CPU_X86>* ptr = (Workqueue<T,LENGTH,CPU_X86>*)LowLevelMemAllocator::alloc(s,MT);
		 return ptr;
	}



	void operator delete[](void *dp) {
		return LowLevelMemAllocator::dealloc(dp,((Workqueue<T,LENGTH,CPU_X86>*)dp)->MT);

	}


	void operator delete(void *dp) {
		return LowLevelMemAllocator::dealloc(dp,((Workqueue<T,LENGTH,CPU_X86>*)dp)->MT);
	}

};
#endif


#endif /* WORKQUEUE_H_ */
