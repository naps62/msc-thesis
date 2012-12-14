/*
 * nthrootsKernel.h
 *
 *  Created on: Nov 22, 2011
 *      Author: amariano
 */

#ifndef NTHKERNEL_H_
#define NTHKERNEL_H_

#include <gamalib/memlib/MemoryPool.h>

class nthrootsKernel : public work {

public:

	smartPtr<double> nthrootsdataR;
	smartPtr<double> nthrootsdataI;
	int startnth;
	int endnth;
	int N;
	int stage;

	__HYBRID__ nthrootsKernel(smartPtr<double> _nthrootsdataR, int _startnth, int _endnth, smartPtr<double> _nthrootsdataI, int _N, int _stage) : nthrootsdataR(_nthrootsdataR), nthrootsdataI(_nthrootsdataI), N(_N), stage(_stage){

		startnth = _startnth;
		endnth = _endnth;

		WORK_TYPE_ID = WORK_NTH | WD_NOT_DICEABLE | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~nthrootsKernel() {
	}

	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		int end = ((int) log2(N*1.0));

		List<work*>* L = new List<work*>(end);

		nthrootsdataR[0] = 1;
		nthrootsdataI[0] = 0;
		nthrootsdataR[1] = 1;
		nthrootsdataI[1] = 0;

		int start = 2, p_stage = 2;

		int __startnth, __endnth;

		for (int k = 0; k < end; k++) {

			__startnth = start;
			__endnth = start+p_stage;

			nthrootsKernel* n = (nthrootsKernel*)LowLevelMemAllocator::sharedMemory(sizeof(nthrootsKernel));
			*n = nthrootsKernel(nthrootsdataR,__startnth,__endnth,nthrootsdataI,N,p_stage);

			(*L)[k] = n;

			start+=start;

			p_stage *= 2;


		}

		return L;
	}

	template<DEVICE_TYPE>
	__HYBRID__ void execute() {

		int middle = (endnth+startnth)/2;

		if(TID > middle) return;
		unsigned tid = TID + startnth;

		double argV = -2*MPI;
		double argV2, argV3;

		for(; tid < middle; tid+=TID_SIZE) {

			argV2 = argV/stage;
			argV3 = argV2 * (tid);

			nthrootsdataR[tid] = cos(argV3);
			nthrootsdataI[tid] = sin(argV3);
		}

	}

};



#endif /* INITARRAY_H_ */
