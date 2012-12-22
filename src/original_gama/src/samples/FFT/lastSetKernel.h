/*
 * lastSetKernel.h
 *
 *  Created on: Dec 1, 2011
 *      Author: amariano
 */

#ifndef LASTSET_H_
#define LASTSET_H_

#define RIGHT 1
#define LEFT 0

#include <gamalib/memlib/MemoryPool.h>

class lastSetKernel : public work {

public:

	smartPtr<double> dataORD;
	smartPtr<double> dataOutput;
	int Pstr;

	smartPtr<double> nthrootsdataR;
	smartPtr<double> nthrootsdataI;

	int wingsize;
	int chunksize;
	int offset;
	//int* B_offset2;
	//int nthreads;
	//int stage;
	int index;
	int butterfliesPerThread;
	int side;

	__HYBRID__ lastSetKernel(smartPtr<double> _dataORD, smartPtr<double> _dataOutput, int _Pstr, smartPtr<double> _nthrootsdataR, smartPtr<double> _nthrootsdataI
			, int _wingsize, int _chunksize, int _offset,/*int* _B_offset1, int _nthreads, int _stage,*/ int _index, int _butterfliesPerThread, int _side){

		dataORD = _dataORD;
		dataOutput = _dataOutput;
		Pstr = _Pstr;
		nthrootsdataR = _nthrootsdataR;
		nthrootsdataI = _nthrootsdataI;
		wingsize = _wingsize;
		chunksize = _chunksize;
		offset = _offset;
		//nthreads = _nthreads;
		//stage = _stage;
		index = _index;
		butterfliesPerThread = _butterfliesPerThread;
		side = _side;

		WORK_TYPE_ID = WORK_LS | WD_NOT_DICEABLE | W_REGULAR | W_WIDE;;
	}

	__HYBRID__ ~lastSetKernel() {
	}


	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		//int N = ARRAY_SIZE;

		int start = 0, end = 8, privateChunksize = ARRAY_SIZE/end;

		List<work*>* L = new List<work*>(end);

		int *offsets = (int*)malloc((end/2)*sizeof(int));

		int k;

		for (k = 0; k < end/2; k++) {

			//printf("START = %d\n",start);

			offsets[k] = (k*chunksize/2)+k*butterfliesPerThread/2;

			lastSetKernel* l = (lastSetKernel*)LowLevelMemAllocator::sharedMemory(sizeof(lastSetKernel));
			*l = lastSetKernel(dataORD,dataOutput,start,nthrootsdataR,nthrootsdataI,wingsize,chunksize,offset,offsets[k],butterfliesPerThread,1);

			(*L)[k] = l;

			start+=(privateChunksize*2);

		}

		for ( ; k < end; k++) {

			//printf("START = %d\n",start);

			lastSetKernel* l = (lastSetKernel*)LowLevelMemAllocator::sharedMemory(sizeof(lastSetKernel));
			*l = lastSetKernel(dataORD,dataOutput,start,nthrootsdataR,nthrootsdataI,wingsize,chunksize,offset,offsets[(k-(end/2))],butterfliesPerThread,0);

			(*L)[k] = l;

			start+=(privateChunksize*2);

		}

		return L;

//		int N = ARRAY_SIZE;
//
//		int start = 0, end = 8;
//
//		int chunksize = N/end;
//
//		int skip = chunksize;
//
//		List<work*>* L = new List<work*>(end);
//
//		for (int k = 0; k < end; k++) {
//
//			lastSetKernel* l = (lastSetKernel*)LowLevelMemAllocator::sharedMemory(sizeof(lastSetKernel));
//			*l = lastSetKernel(dataORD,dataOutput,start,nthrootsdataR,nthrootsdataI,wingsize,chunksize,offset,k*chunksize/2,butterfliesPerThread);
//
//			(*L)[k] = l;
//
//			start+=skip;
//
//		}
//
//		return L;
	}

	template<DEVICE_TYPE>
	__HYBRID__ void execute() {

		if(side==RIGHT){

			int loopEnd = Pstr + butterfliesPerThread*2;

			double _nthrootreal, _nthrootimgy, _temp1R, _temp1I;

			int _skip = wingsize*2+index;

			int summedWingsize;

			if(TID >= butterfliesPerThread*2) return;
			unsigned tid = Pstr + TID*2;

			int it = TID;

			//Loop to perform semi-butterflies;
			for( ;tid<loopEnd; tid+=TID_SIZE*2 ){

				summedWingsize = tid + (wingsize<<1);

				_nthrootreal = nthrootsdataR[_skip+it];
				_nthrootimgy = nthrootsdataI[_skip+it];

				_temp1R = (dataORD[summedWingsize] * _nthrootreal) - (dataORD[summedWingsize+1] * _nthrootimgy);
				_temp1I = (dataORD[summedWingsize] * _nthrootimgy) + (dataORD[summedWingsize+1] * _nthrootreal);

				dataOutput[summedWingsize] = dataORD[tid] - _temp1R;
				dataOutput[summedWingsize+1] = dataORD[tid+1] - _temp1I;

				it += TID_SIZE;
			}

		}
		else{

			int loopEnd = Pstr + butterfliesPerThread*2;

			double _nthrootreal, _nthrootimgy, _temp1R, _temp1I;

			int _skip = wingsize*2+index;

			int summedWingsize;

			if(TID >= butterfliesPerThread*2) return;
			unsigned tid = Pstr + TID*2;

			int it = TID;

			//Loop to perform semi-butterflies;
			for( ;tid<loopEnd; tid+=TID_SIZE*2 ){

				summedWingsize = tid - (wingsize<<1);

				_nthrootreal = nthrootsdataR[_skip+it];
				_nthrootimgy = nthrootsdataI[_skip+it];

				_temp1R = (dataORD[tid] * _nthrootreal) - (dataORD[tid+1] * _nthrootimgy);
				_temp1I = (dataORD[tid] * _nthrootimgy) + (dataORD[tid+1] * _nthrootreal);

				//Outplace operation: cannot be += ...
				dataOutput[summedWingsize] = dataORD[summedWingsize] + _temp1R;
				dataOutput[summedWingsize+1] = dataORD[summedWingsize+1] + _temp1I;

				it += TID_SIZE;
			}


		}

	}


















//CPU ONLY:

//		if(side==RIGHT){
//
//			int loopEnd = Pstr + butterfliesPerThread, it = 0;
//
//			double _nthrootreal, _nthrootimgy, _temp1R, _temp1I;
//
//			int _skip = wingsize*2+index;
//
//			int summedWingsize;
//
//			if(TID > butterfliesPerThread) return;
//			unsigned tid = TID + Pstr;
//
//			//Loop to perform semi-butterflies;
//			for( ;tid<loopEnd; tid+=TID_SIZE ){
//
//				summedWingsize = tid + (wingsize<<1) + it;
//
////				printf("\n");
////				printf("(%d)\n",summedWingsize);
////				printf("(%d) dataORD[%d] = %f\n",summedWingsize,(summedWingsize+2*TID)+TID,dataORD[(summedWingsize+2*TID)+TID]);
////				printf("(%d) dataORD[%d] = %f\n",summedWingsize,(summedWingsize+2*TID)+TID+1,dataORD[(summedWingsize+2*TID)+TID+1]);
////				printf("(%d) nthrootsdataR[%d]\n",summedWingsize,_skip+it);
////				printf("(%d) nthrootsdataI[%d]\n",summedWingsize,_skip+it);
//
//				_nthrootreal = nthrootsdataR[_skip+it];
//				_nthrootimgy = nthrootsdataI[_skip+it];
//
////				if(tid==0){
////					printf("_tmp1R = %f\n",_temp1R);
////					printf("_nthrootreal = %f\n",_nthrootreal);
////					printf("_nthrootimg = %f\n",_nthrootimgy);
////
////				}
//
//				_temp1R = (dataORD[(summedWingsize+2*TID)+TID] * _nthrootreal) - (dataORD[(summedWingsize+2*TID)+TID+1] * _nthrootimgy);
//				_temp1I = (dataORD[(summedWingsize+2*TID)+TID] * _nthrootimgy) + (dataORD[(summedWingsize+2*TID)+TID+1] * _nthrootreal);
//
//				dataOutput[summedWingsize] = dataORD[tid+it] - _temp1R;
//				dataOutput[summedWingsize+1] = dataORD[tid+it+1] - _temp1I;
//
//				it++;
//			}
//
//		}
//		else{
//
//			int loopEnd = butterfliesPerThread + Pstr, it = 0;
//
//			double _nthrootreal, _nthrootimgy, _temp1R, _temp1I;
//
//			int _skip = wingsize*2+index;
//
//			int summedWingsize;
//
//			if(TID > butterfliesPerThread) return;
//			unsigned tid = TID + Pstr;
//
//			//Loop to perform semi-butterflies;
//			for( ;tid<loopEnd; tid+=TID_SIZE ){
//
//				summedWingsize = tid - (wingsize<<1) + it;
//
//				_nthrootreal = nthrootsdataR[_skip+it];
//				_nthrootimgy = nthrootsdataI[_skip+it];
//
//				_temp1R = (dataORD[tid+it] * _nthrootreal) - (dataORD[tid+it+1] * _nthrootimgy);
//				_temp1I = (dataORD[tid+it] * _nthrootimgy) + (dataORD[tid+it+1] * _nthrootreal);
//
//				//Outplace operation: cannot be += ...
//				dataOutput[summedWingsize] = dataORD[summedWingsize] + _temp1R;
//				dataOutput[summedWingsize+1] = dataORD[summedWingsize+1] + _temp1I;
//
//				it++;
//			}
//
//
//		}
//
//	}



};

#endif /* INITARRAY_H_ */
