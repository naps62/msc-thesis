/*
 * MemotyPool.h
 *
 *  Created on: May 3, 2012
 *      Author: jbarbosa
 */

#ifndef MEMOTYPOOL_H_
#define MEMOTYPOOL_H_



template<typename T>
class MemoryPool {

	T* ptrs;
	unsigned int next;

public:
	__DEVICE__
	MemoryPool(size_t elements) : next(0) {
		ptrs = (T*)(LowLevelMemAllocator::alloc(sizeof(T)*elements,SHARED));
	}

	__DEVICE__
	virtual ~MemoryPool() {
		//LowLevelMemAllocator::dealloc(ptrs,SHARED);
	}

	__DEVICE__
	T* pool() {
		return &ptrs[next++];
	}
};



#endif /* MEMOTYPOOL_H_ */
