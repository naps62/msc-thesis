/*
 * InitArray.h
 *
 *  Created on: Nov 22, 2011
 *      Author: amariano
 */

#ifndef BITTLB_H_
#define BITTLB_H_

#include <gamalib/memlib/MemoryPool.h>

class bitTable : public work {

public:

	smartPtr<int> table;

	int start, end;

	int N;
	unsigned int logN;

	__HYBRID__ bitTable(smartPtr<int> _data, int _start, int _end, int _N){
		table = _data;

		start = _start;
		end = _end;

		N = _N;

		logN = log2((float)N*2);

		WORK_TYPE_ID = WORK_TBL | WD_NOT_DICEABLE | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~bitTable() {
	}

	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		List<work*>* L;

		int chunk, items;

		if(N>number){
			L = new List<work*>(number);
			chunk = (N*2 / number);
			items = number;
		}
		else{
			L = new List<work*>(1);
			chunk = N*2;
			items = 1;
		}

		int start = 0, end = chunk;

		for (int k = 0; k < items; k++) {

			bitTable* s = (bitTable*)LowLevelMemAllocator::sharedMemory(sizeof(bitTable));
			*s = bitTable(table,start,end,N);

			(*L)[k] = s;

			start = end;
			end += chunk;

		}

		return L;
	}

	template<DEVICE_TYPE>
	__HYBRID__ void execute() {

		if(TID > ((end-start))) return;
		unsigned tid = (TID + start);

		unsigned int nforward, nreversed;
		unsigned int count;

		for( ; (tid)<(end); tid+=TID_SIZE*2){

			nreversed = tid;
			count = logN-1;

			for(nforward=(tid)>>1; nforward; nforward>>=1){

			   nreversed <<= 1;
			   nreversed |= nforward & 1; 	// give LSB of nforward to nreversed
			   count--;

			}

			nreversed <<=count; 			// compensation for missed iterations
			nreversed &= N-1;   			// clear all bits more significant than N-1

			table[tid] = (nreversed*2);
			table[tid+1] = (nreversed*2)+1;

		}

	}
};



#endif /* INITARRAY_H_ */
