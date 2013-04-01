/*
 * saxpy.h
 *
 *  Created on: Dec 14, 2012
 *      Author: naps62
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <gamalib/gamalib.h>
#include <gama_ext/vector.h>

class kernel : public work {

public:
	gama::vector<int> arr;
	unsigned int lower;
	unsigned int upper;

	__HYBRID__ kernel(gama::vector<int> _arr, unsigned int lo, unsigned int hi)
	: arr(_arr), lower(lo), upper(hi)
	{
		WORK_TYPE_ID = WORK_KERNEL | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~kernel() { }


	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {
		unsigned range = (upper - lower);
		unsigned number_of_units = range / number;

		if (number_of_units == 0) {
			number_of_units = 1;
			number = range;
		}

		unsigned start = lower;

		List<work*>* L = new List<work*>(number);
		for(unsigned k = 0; k < number; ++k) {
			kernel* s = (kernel*)LowLevelMemAllocator::sharedMemory(sizeof(kernel));
			*s = kernel(arr, start, start + number_of_units);
			(*L)[k] = s;
			start += number_of_units;
		}

		return L;
	}


	template<DEVICE_TYPE>
	__DEVICE__ void execute() {
		if (TID > (upper-lower)) return;

		unsigned long tid = TID + lower;
		for(; tid < upper; tid += TID_SIZE) {
			arr[tid] = arr[tid] + 1;
		}
	}
};

#endif /* KERNEL_H */
