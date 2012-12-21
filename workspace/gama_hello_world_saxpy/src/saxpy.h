/*
 * saxpy.h
 *
 *  Created on: Dec 14, 2012
 *      Author: naps62
 */

#ifndef SAXPY_H_
#define SAXPY_H_

#include <gama.h>
#include <float.h>

class saxpy : public work {

public:
	smartPtr<float> R;
	smartPtr<float> X;
	smartPtr<float> Y;

	float alpha;

	unsigned int LENGTH;
	unsigned int lower;
	unsigned int upper;

	__HYBRID__ saxpy(
			smartPtr<float> _R, smartPtr<float> _X, smartPtr<float> _Y,
			float _alpha, unsigned _LENGTH, unsigned _lower, unsigned _upper)
		: R(_R), X(_X), Y(_Y),
		  alpha(_alpha), LENGTH(_LENGTH), lower(_lower), upper(_upper)
	{
		WORK_TYPE_ID = WORK_SAXPY | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~saxpy() { }


	template<DEVICE_TYPE> __DEVICE__ List<work*>* dice(unsigned int &number) {
		unsigned range = (upper - lower);
		unsigned number_of_units = range / number;

		if (number_of_units == 0) {
			number_of_units = 1;
			number = range;
		}

		unsigned start = lower;

		List<work*>* L = new List<work*>(number);
		for(unsigned k = 0; k < number; ++k) {
//			saxpy* s = new saxpy(R, X, Y, alpha, LENGTH, start, start + number_of_units);
			saxpy* s = (saxpy*)LowLevelMemAllocator::sharedMemory(sizeof(saxpy));
			*s = saxpy(R, X, Y, alpha, LENGTH, start, start + number_of_units);
			(*L)[k] = s;
			start += number_of_units;
		}

		return L;
	}


	template<DEVICE_TYPE> __DEVICE__ void execute() {
		if (TID > (upper-lower)) return;

		unsigned long tid = TID + lower;
		for(; tid < upper; tid += TID_SIZE) {
			R[tid] = X[tid] * alpha + Y[tid];
		}
	}

	//TODO figure this out
//	std::vector<pointerInfo>* toCacheR(){
//		std::vector<pointerInfo>* L = new std::vector<pointerInfo>(0);
//		L->push_back(pointerInfo(X.getPtr(lower),(upper-lower)*sizeof(float)));
//		L->push_back(pointerInfo(Y.getPtr(lower),(upper-lower)*sizeof(float)));
//		return L;
//	}
//
//
//
//	std::vector<pointerInfo>* toCacheW(){
//		std::vector<pointerInfo>* L = new std::vector<pointerInfo>(0);
//		return L;
//	}
};

#endif /* SAXPY_H_ */
