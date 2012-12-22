/*
 * InitArray.h
 *
 *  Created on: Nov 22, 2011
 *      Author: amariano
 */

#ifndef BITKERNEL_H_
#define BITKERNEL_H_

#include <gamalib/memlib/MemoryPool.h>

class bitKernel : public work {

public:

	smartPtr<int> tbl;
	smartPtr<double> data;

	int initdata;
	int enddata;

	smartPtr<double> dataORD;

	int initORD;
	int endORD;

	int* offsetArray;
	int chunksize;
	int id;
	int offset;
	int worksize;
	int N;
	int workers;
	int logN;

	__HYBRID__ bitKernel(smartPtr<int> _tbl, smartPtr<double> _data, int a, int b, smartPtr<double> _dataORD,
			int c, int d, int* _offsetArray, int _chunksize, int _id, int _offset, int _N, int _workers){

		tbl = _tbl;
		data = _data;

		initdata = a;
		enddata = b;

		dataORD = _dataORD;

		initORD = c;
		endORD = d;

		offsetArray = _offsetArray;
		chunksize = _chunksize;
		id = _id;
		offset = _offset;
		N = _N;

		logN = log2((float)_N);

		workers = _workers;

		WORK_TYPE_ID = WORK_BRK | WD_NOT_DICEABLE | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~bitKernel() {
	}

	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		List<work*>* L = new List<work*>(workers);

		int skip = (ARRAY_SIZE / workers);
		int start=0;

		for (int k = 0; k < workers; k++) {

			bitKernel* s = (bitKernel*)LowLevelMemAllocator::sharedMemory(sizeof(bitKernel));
			//*s = bitKernel(tbl,data,0,dataORD._size,dataORD,start,(start+skip),NULL,skip,k,offsetArray[k],N,workers);

			(*L)[k] = s;

			start+=skip;

		}

		return L;
	}

	template<DEVICE_TYPE>
	__HYBRID__ void execute() {

// Funciona (Activo):
//
		if(TID > ((endORD-initORD))) return;
		unsigned tid = (TID + initORD);

//		unsigned int bits = logN;
//		unsigned int nforward, nreversed;
//		unsigned int count;

		for( ; (tid)<(endORD); tid+=TID_SIZE){

//			nreversed = tid;
//			count = bits-1;
//
//			for(nforward=(tid)>>1; nforward; nforward>>=1){
//
//			   nreversed <<= 1;
//			   nreversed |= nforward & 1; 	// give LSB of nforward to nreversed
//			   count--;
//
//			}
//
//			nreversed <<=count; 			// compensation for missed iterations
//			nreversed &= N-1;   			// clear all bits more significant than N-1

			dataORD[tid*2] = data[tbl[tid*2]];//data[nreversed*2];//
			dataORD[tid*2+1] = data[tbl[tid*2+1]];//data[nreversed*2+1];//

		}

//		if(TID > (endORD-initORD)) return;
//		unsigned tid = TID + initORD;
//
//		int j=0,k;
//
//		for(; tid < endORD; tid+=TID_SIZE) {
//
//			if(tid==initORD) j=0;
//			else{
//
//				for(int aux=0;aux<(tid-initORD) ;aux++){
//
//					k=N/2;
//					while(k<=j){
//						j = j-k;
//						k = k/2;
//					}
//					j=j+k;
//
//					if(TID==0) break;
//
//				}
//
//			}
//
//			dataORD[tid] = data[j+offset];
//			dataORD[tid+1] = data[j+offset+1];
//
//		}

//		unsigned int i, forward, rev, zeros;
//		unsigned int nodd, noddrev;        // to hold bitwise negated or odd values
//		unsigned int N, halfn, quartn, nmin1;
//		double temp;
//
//		N = 1<<logN;
//		halfn = N>>1;            // frequently used 'constants'
//		quartn = N>>2;
//		nmin1 = N-1;
//
//		forward = halfn;         // variable initialisations
//		rev = 1;
//
//		for(i=quartn; i; i--)    // start of bitreversed permutation loop, N/4 iterations
//		    {
//
//		     // Gray code generator for even values:
//
//		     nodd = ~i;                                  // counting ones is easier
//		     for(zeros=0; nodd&1; zeros++) nodd >>= 1;   // find trailing zero's in i
//		     forward ^= 2 << zeros;                      // toggle one bit of forward
//		     rev ^= quartn >> zeros;                     // toggle one bit of reversed
//
//
//		        if(forward<rev)                  // swap even and ~even conditionally
//		        {
//		            temp = dataORDR[forward];
//		            dataORDR[forward] = dataORDR[rev];
//		            dataORDR[rev] = temp;
//
//		            nodd = nmin1 ^ forward;           // compute the bitwise negations
//		            noddrev = nmin1 ^ rev;
//
//		            temp = dataORDR[nodd];                // swap bitwise-negated pairs
//		            dataORDR[nodd] = dataORDR[noddrev];
//		            dataORDR[noddrev] = temp;
//		        }
//
//		        nodd = forward ^ 1;                   // compute the odd values from the even
//		        noddrev = rev ^ halfn;
//
//		        temp = dataORDR[nodd];                    // swap odd unconditionally
//		        dataORDR[nodd] = dataORDR[noddrev];
//		        dataORDR[noddrev] = temp;
//		    }

//		unsigned int bits = logN;
//		unsigned int NMIN1 = N-1;
//		unsigned int n;                    // index
//		unsigned int count;
//		unsigned int nforward, nreversed;
//		unsigned int notn, notnreversed;   // for bitwise negated values
//		unsigned int temp;                 // to store x[n] during swap
//
//		for(n=initORD; n<endORD>>1; n++)   // only N/4 iterations, extra increment within loop
//		{
//		    count = bits-1;
//		    nreversed = n;
//
//		    for(nforward=n>>1; nforward; nforward>>=1) // reverses the bit-pattern of n
//		    {
//		       nreversed <<= 1;
//		       nreversed |= nforward & 1;   // give LSB of nforward to nreversed
//		       count--;
//		    }
//		    nreversed <<= count;       // compensation for skipped iterations
//		    nreversed &= NMIN1;        // cut off all bits more significant than N-1
//
//		    if(n<nreversed)            // for even n, swap conditionally
//		    {
//		       dataORDR[n] = dataR[nreversed];
//		       dataORDR[nreversed] = dataR[n];
//
//		       notn = NMIN1 ^ n;                  // compute bitwise negations
//		       notnreversed = NMIN1 ^ nreversed;
//
//		       dataORDR[n] = dataR[nreversed];
//		       dataORDR[nreversed] = dataR[nreversed];
//		    }
//
//		    // odd n and nreversed can be swapped unconditionally
//
//		    n++;                                 // attention, extra increment!
//		    nreversed |= N>>1;                   // set highest bit in nreversed
//
//		       dataORDR[n] = dataR[nreversed];
//		       dataORDR[nreversed] = dataR[nreversed];
//		}

	}
};



#endif /* INITARRAY_H_ */
