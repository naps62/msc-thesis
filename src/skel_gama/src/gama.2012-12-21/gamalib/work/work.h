/*
 * work.h
 *
 *  Created on: Apr 3, 2012
 *      Author: jbarbosa
 */

#ifndef WORK_H_
#define WORK_H_


#include <config/common.h>
#include <gamalib/memlib/smartpointer.h>
#include <gamalib/Datastructures/List.h>
#include <vector>

struct pointerInfo{
	void* ptr;
	size_t lenght;
	pointerInfo() : ptr(NULL), lenght(0) {}
	pointerInfo(void* p, size_t l) : ptr(p), lenght(l) {}

};

class work {

public:

	unsigned long WORK_TYPE_ID;

	__HYBRID__ work(WORK_TYPE W_T_ID) :  WORK_TYPE_ID(W_T_ID) {
	};

	__HYBRID__ work() :  WORK_TYPE_ID(WORK_NONE) {
	};


	__HYBRID__ ~work() {
	};

	__HYBRID__ __forceinline unsigned int getWorkTypeID() {
		return FILTER_WORK(WORK_TYPE_ID);
	}

	template<DEVICE_TYPE> __HYBRID__ List<work*>* dice(unsigned int &number) {

		List<work*>* L = new List<work*>(number);

		for(int i=0; i < number; i++)
			(*L)[i] = new work();
		return L;
	}

	template<DEVICE_TYPE> __HYBRID__ void execute() {
	}


	std::vector<pointerInfo>* toCacheR() {
		return new std::vector<pointerInfo>(0);
	}

	std::vector<pointerInfo>* toCacheW() {
		return new std::vector<pointerInfo>(0);
	}

//	__HYBRID__
//	virtual unsigned int length() {return 0;};


	__DEVICE__
	void *operator new(size_t s) {
#ifndef __CUDACC__
		return LowLevelMemAllocator::alloc(s,SHARED);
#else
		return malloc(s);
#endif
	}

	__DEVICE__
	void *operator new[](size_t s) {
		return LowLevelMemAllocator::alloc(s,SHARED);
	}


	__DEVICE__
	void operator delete[](void *dp) {
		LowLevelMemAllocator::dealloc(dp,SHARED);
	}

	__DEVICE__
	void operator delete(void *dp) {
#ifndef __CUDACC__
		LowLevelMemAllocator::dealloc(dp,SHARED);
#else
		free(dp);
#endif
	}

};



#endif /* WORK_H_ */
