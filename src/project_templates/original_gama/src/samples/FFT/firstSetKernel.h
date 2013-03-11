/*
 * firstSetKernel.h
 *
 *  Created on: Dec 1, 2011
 *      Author: amariano
 */

#ifndef FIRSTSET_H_
#define FIRSTSET_H_

#include <gamalib/memlib/MemoryPool.h>

class firstSetKernel : public work {

public:

	smartPtr<double> dataORD;
	smartPtr<double> nthrootsdataR;
	smartPtr<double> nthrootsdataI;

	int Pstr;
	int Pend;

	int wingsize;
	int chunksize;
	int butterflies;
	int side;

	__HYBRID__ firstSetKernel(smartPtr<double> _dataORD, int _Pstr, int _Pend, smartPtr<double> _nthrootsdataR, smartPtr<double> _nthrootsdataI,
			int _wingsize, int _chunksize, int _butterflies, int _side/*, int _nthreads, int _stage, int _ID*/){
		dataORD = _dataORD;
		nthrootsdataR = _nthrootsdataR;
		nthrootsdataI = _nthrootsdataI;

		Pstr = _Pstr;
		Pend = _Pend;

		wingsize = _wingsize;
		chunksize = _chunksize;
		butterflies = _butterflies;
		side = _side;

		WORK_TYPE_ID = WORK_FS | WD_NOT_DICEABLE | W_REGULAR | W_WIDE;;
	}

	__HYBRID__ ~firstSetKernel() {
	}

	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		int start = 0, end = ARRAY_SIZE / chunksize, butterflies = (ARRAY_SIZE/2) / end ;

		List<work*>* L = new List<work*>(end);

		for (int k = 0; k < end; k++) {

			firstSetKernel* f = (firstSetKernel*)LowLevelMemAllocator::sharedMemory(sizeof(firstSetKernel));
			*f = firstSetKernel(dataORD,start,start+(chunksize*2),nthrootsdataR,nthrootsdataI,wingsize,chunksize,butterflies-1,0 );

			(*L)[k] = f;

			//printf("START = %d\n",start);

			start+=chunksize*2;


		}

		return L;
	}

	template<DEVICE_TYPE>
	__HYBRID__ void execute() {

//Concurrency issues...
//Generic (CPU || GPU) code (working):

		int skip = wingsize*4, doubledWingsize = wingsize*2, it = 0;

		double temp1R, temp1I, temp2R, temp2I, _nthrootreal, _nthrootimg, _dataORDreal, _dataORDimg;

		if(TID > butterflies) return;

		int i = TID;
		int index = Pstr + ((i/wingsize) * skip) + (i%wingsize)*2;

		//printf("Start = %d Thread %d starts in %d \n", Pstr, i, index );

		for( ;index < Pend;  ){

			it = i%wingsize;

			//	printf("index =%d\n", index);

			temp1R = dataORD[index];
			temp1I = dataORD[index+1];

			//----------------------
			_nthrootreal = nthrootsdataR[skip+it];
			_nthrootimg = nthrootsdataI[skip+it];

			_dataORDreal = dataORD[index+doubledWingsize];
			_dataORDimg = dataORD[index+doubledWingsize+1];

			//>>>>>complexMULT
			temp2R = (_dataORDreal * _nthrootreal) - (_dataORDimg * _nthrootimg);
			temp2I = (_dataORDreal * _nthrootimg) + (_dataORDimg * _nthrootreal);
			//<<<<<complexMULT

			//>>>>>complexSUM
			dataORD[index] += temp2R;
			dataORD[index+1] += temp2I;
			//>>>>>complexSUM

			//>>>>>complexSUB
			dataORD[index+doubledWingsize] = temp1R - temp2R;
			dataORD[index+doubledWingsize+1] = temp1I - temp2I;
			//>>>>>complexSUB

			//it++;

			i += TID_SIZE;
			index = Pstr + ((i/wingsize) * skip) + (i%wingsize)*2;

		}
	}

///* CPU part:
/*
		int k, summedWingsize, skip = wingsize<<2, end, summedSkipsize;

		for(int i=0;i<(chunksize<<1);i+=skip){

			k=0, end = (Pstr+i+(wingsize<<1));

			if(TID > (wingsize<<1)) return;
			unsigned tid = TID + Pstr + i;

			for( ;tid<end;tid+=(TID_SIZE*2) ){

				double temp1R, temp1I, temp2R, temp2I;

				summedSkipsize = skip+k;
				summedWingsize = end+k;

				temp1R = dataORD[tid];
				temp1I = dataORD[tid+1];

				//----------------------
				double _nthrootreal = nthrootsdataR[summedSkipsize];
				double _nthrootimg = nthrootsdataI[summedSkipsize];

				//>>>>>complexMULT
				temp2R = (dataORD[summedWingsize] * _nthrootreal) - (dataORD[summedWingsize+1] * _nthrootimg);
				temp2I = (dataORD[summedWingsize] * _nthrootimg) + (dataORD[summedWingsize+1] * _nthrootreal);
				//<<<<<complexMULT

				//>>>>>complexSUM
				dataORD[tid] = temp1R + temp2R;
				dataORD[tid+1] = temp1I + temp2I;
				//>>>>>complexSUM

				//>>>>>complexSUB
				dataORD[summedWingsize] = temp1R - temp2R;
				dataORD[summedWingsize+1] = temp1I - temp2I;
				//>>>>>complexSUB

				k++;

			}
		}
	}
*/
//*/

};

#endif /* INITARRAY_H_ */

#ifndef FIRSTSET_H_
#define FIRSTSET_H_

#include <gamalib/memlib/MemoryPool.h>

class firstSetKernel : public work {

public:

	smartPtr<double> dataORD;
	smartPtr<double> nthrootsdataR;
	smartPtr<double> nthrootsdataI;

	int Pstr;
	int Pend;

	int wingsize;
	int chunksize;

	__HYBRID__ firstSetKernel(smartPtr<double> _dataORD, int _Pstr, int _Pend, smartPtr<double> _nthrootsdataR, smartPtr<double> _nthrootsdataI, int _wingsize, int _chunksize/*, int _nthreads, int _stage, int _ID*/){
		dataORD = _dataORD;
		nthrootsdataR = _nthrootsdataR;
		nthrootsdataI = _nthrootsdataI;

		Pstr = _Pstr;
		Pend = _Pend;

		wingsize = _wingsize;
		chunksize = _chunksize;

		WORK_TYPE_ID = WORK_FS | WD_NOT_DICEABLE | W_REGULAR | W_WIDE;;
	}

	__HYBRID__ ~firstSetKernel() {
	}

	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		int start = 0, end = ARRAY_SIZE / chunksize;

		List<work*>* L = new List<work*>(end);

		for (int k = 0; k < end; k++) {

			firstSetKernel* f = (firstSetKernel*)LowLevelMemAllocator::sharedMemory(sizeof(firstSetKernel));
			*f = firstSetKernel(dataORD,start,start+(chunksize*2),nthrootsdataR,nthrootsdataI,wingsize,chunksize);

			(*L)[k] = f;

			start+=chunksize*2;

		}

		return L;
	}

	template<DEVICE_TYPE>
	__HYBRID__ void execute() {

		int k, summedWingsize, skip = wingsize<<2, end, summedSkipsize;

		for(int i=0;i<(chunksize<<1);i+=skip){

			k=0, end = (Pstr+i+(wingsize<<1));

			if(TID > (wingsize<<1)) return;
			unsigned tid = TID + Pstr + i;

			for( ;tid<end;tid+=(TID_SIZE*2) ){

				double temp1R, temp1I, temp2R, temp2I;

				summedSkipsize = skip+k*TID_SIZE+TID;
				summedWingsize = end+k*TID_SIZE+TID;

				temp1R = dataORD[tid];
				temp1I = dataORD[tid+1];

				//----------------------
				double _nthrootreal = nthrootsdataR[summedSkipsize];
				double _nthrootimg = nthrootsdataI[summedSkipsize];

				//>>>>>complexMULT
				temp2R = (dataORD[summedWingsize] * _nthrootreal) - (dataORD[summedWingsize+1] * _nthrootimg);
				temp2I = (dataORD[summedWingsize] * _nthrootimg) + (dataORD[summedWingsize+1] * _nthrootreal);
				//<<<<<complexMULT

				//>>>>>complexSUM
				dataORD[tid] = temp1R + temp2R;
				dataORD[tid+1] = temp1I + temp2I;
				//>>>>>complexSUM

				//>>>>>complexSUB
				dataORD[summedWingsize] = temp1R - temp2R;
				dataORD[summedWingsize+1] = temp1I - temp2I;
				//>>>>>complexSUB

				k++;

			}
		}
	}

};

#endif /* INITARRAY_H_ */


