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

struct RXY {
	smartPtr<float> r;
	smartPtr<float> x;
	smartPtr<float> y;

	RXY(RXY& _rxy)
	: r(_rxy.r), x(_rxy.x), y(_rxy.y)  {

	}

	RXY(int N) {
		r = smartPtr<float>(sizeof(float)*N);
		x = smartPtr<float>(sizeof(float)*N);
		y = smartPtr<float>(sizeof(float)*N);
	}

	__DEVICE__
	void calc(unsigned long i, float alpha) {
		r[i] = x[i] * alpha + y[i];
	}
};

class saxpy : public work {

public:
	RXY rxy;

	float alpha;

	unsigned int LENGTH;
	unsigned int lower;
	unsigned int upper;

	__HYBRID__ saxpy(RXY& _rxy,
			float _alpha, unsigned _LENGTH, unsigned _lower, unsigned _upper)
		: rxy(_rxy),
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
			saxpy* s = (saxpy*)LowLevelMemAllocator::sharedMemory(sizeof(saxpy));
			*s = saxpy(rxy, alpha, LENGTH, start, start + number_of_units);
			(*L)[k] = s;
			start += number_of_units;
		}

		return L;
	}


	template<DEVICE_TYPE> __DEVICE__ void execute() {
		if (TID > (upper-lower)) return;

		unsigned long tid = TID + lower;
		for(; tid < upper; tid += TID_SIZE) {
			rxy.calc(tid, alpha);
			//rxy.r[tid] = rxy.x[tid] * alpha + rxy.y[tid];
		}
	}

};

#endif /* SAXPY_H_ */
