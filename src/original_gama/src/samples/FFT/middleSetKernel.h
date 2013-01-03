/*
 * middleSetKernel.h
 *
 *  Created on: Dec 1, 2011
 *      Author: amariano
 */

#ifndef MIDDLESET_H_
#define MIDDLESET_H_

#include <gamalib/memlib/MemoryPool.h>

#define RIGHT 1
#define LEFT 0

class middleSetKernel : public work {

public:

	smartPtr<double> dataORD;

	smartPtr<double> nthrootsdataR;
	smartPtr<double> nthrootsdataI;
	int Pstr;
	int wingsize;
	int chunksize;
	int threadsPerBlock;
	int butterfliesPerThread;
	int index;
	int side;

	__HYBRID__ middleSetKernel(smartPtr<double> _dataORD, int start, smartPtr<double> _nthrootsdataR, smartPtr<double> _nthrootsdataI,
			                int _wingsize, int _chunksize, int _butterfliesPerThread, int _threadsPerBlock, int _index, int _side){
		dataORD = _dataORD;

		nthrootsdataR = _nthrootsdataR;
		nthrootsdataI = _nthrootsdataI;
		Pstr = start;
		wingsize = _wingsize;
		chunksize = _chunksize;
		butterfliesPerThread = _butterfliesPerThread;
		threadsPerBlock = _threadsPerBlock;
		index = _index;
		side = _side;

		WORK_TYPE_ID = WORK_MS | WD_NOT_DICEABLE | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~middleSetKernel() {
	}

	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		int start = 0, end = ARRAY_SIZE / chunksize;

		int final, block;

		int blockSize = threadsPerBlock*chunksize;

		List<work*>* L = new List<work*>(end);

		for (int k = 0; k < end; k++) {

			final = (k%threadsPerBlock) * butterfliesPerThread;
			block = k/threadsPerBlock;

			start = (2*blockSize*block+(k%threadsPerBlock*butterfliesPerThread));

			//printf("start = %d\n",start );

			middleSetKernel* m = (middleSetKernel*)LowLevelMemAllocator::sharedMemory(sizeof(middleSetKernel));
			*m = middleSetKernel(dataORD,start,nthrootsdataR,nthrootsdataI,wingsize,chunksize,butterfliesPerThread,threadsPerBlock,final/2,0);
			(*L)[k] = m;
		}

		return L;

	}

	template<DEVICE_TYPE>
		__HYBRID__ void execute() {

//Concurrency issues...
//Generic (CPU || GPU) code (working):

			int doubledWingsize = wingsize<<1, summedWingsize = doubledWingsize+index;

			int _i1, _i2, _i3;

			double temp1R, temp1I, temp2R, temp2I, _nthrootreal, _nthrootimg;

			if(TID >= butterfliesPerThread) return;
			unsigned tid = Pstr + TID*2;
//
//			int pr = TID;
//
//			printf("my tid is = %d and my TID is = %d and my Pstr = %d \n",tid,pr,Pstr );
//			printf("butterfliesPerThread = %d\n",butterfliesPerThread);
//			printf("\n");

			unsigned _tidSum = TID;

			for( ;tid<(butterfliesPerThread+Pstr); tid+=TID_SIZE*2, _tidSum+=TID_SIZE ){

				_i1 = doubledWingsize+tid;
				_i2 = _i1 + 1;
				_i3 = summedWingsize+_tidSum;

				temp1R = dataORD[tid];
				temp1I = dataORD[tid+1];

				_nthrootreal = nthrootsdataR[_i3];
				_nthrootimg = nthrootsdataI[_i3];

				//>>>>>complexMULT
				/*if(tid%2==0)*/ temp2R = ((dataORD[_i1] * _nthrootreal)) - ((dataORD[_i2] * _nthrootimg));
				/*else*/ temp2I = ((dataORD[_i1] * _nthrootimg)) + ((dataORD[_i2] * _nthrootreal));
				//<<<<<complexMULT

				//>>>>>complexSUM
				/*if(tid%2==0)*/ dataORD[tid] = temp1R + temp2R;
				/*else*/ dataORD[tid+1] = temp1I + temp2I;
				//>>>>>complexSUM

				//>>>>>complexSUB
				/*if(tid%2==0)*/ dataORD[_i1] = temp1R - temp2R;
				/*else*/ dataORD[_i2] = temp1I - temp2I;
				//>>>>>complexSUB

			}
	}

//	int k = index, it = 0, doubledWingsize = wingsize<<1;
//
//				int _i1, _i2, _i3;
//
//				double temp1R, temp1I, temp2R, temp2I, _nthrootreal, _nthrootimg;
//
//				if(TID > butterfliesPerThread) return;
//				unsigned tid = TID + Pstr;
//
//				for(;tid<(butterfliesPerThread+Pstr);tid+=TID_SIZE*2){
//
//						_i1 = doubledWingsize+tid;
//						_i2 = _i1 + 1;
//						_i3 = doubledWingsize+k+it*TID_SIZE+TID;
//
//						temp1R = dataORD[tid];
//						temp1I = dataORD[tid+1];
//
//						_nthrootreal = nthrootsdataR[_i3];
//						_nthrootimg = nthrootsdataI[_i3];
//
//						//>>>>>complexMULT
//						temp2R = ((dataORD[_i1] * _nthrootreal)) - ((dataORD[_i2] * _nthrootimg));
//						temp2I = ((dataORD[_i1] * _nthrootimg)) + ((dataORD[_i2] * _nthrootreal));
//						//<<<<<complexMULT
//
//						//>>>>>complexSUM
//						dataORD[tid] = temp1R + temp2R;
//						dataORD[tid+1] = temp1I + temp2I;
//						//>>>>>complexSUM
//
//						//>>>>>complexSUB
//						dataORD[_i1] = temp1R - temp2R;
//						dataORD[_i2] = temp1I - temp2I;
//						//>>>>>complexSUB
//
//						it++;
//				}
//
//
//		}

};

#endif /* INITARRAY_H_ */
