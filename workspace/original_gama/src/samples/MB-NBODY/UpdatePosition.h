/*
 * UpdatePosition.h
 *
 *  Created on: Jul 24, 2012
 *      Author: ricardo
 */

#ifndef UPDATEPOS_H_
#define UPDATEPOS_H_


class UpdatePosition : public work {
public:

	smartPtr<float> gmass;
	smartPtr<float> gposx;
	smartPtr<float> gposy;
	smartPtr<float> gposz;
	smartPtr<float> gvelx;
	smartPtr<float> gvely;
	smartPtr<float> gvelz;
	smartPtr<float> gaccx;
	smartPtr<float> gaccy;
	smartPtr<float> gaccz;

	unsigned long lower;
	unsigned long upper;
	unsigned long step;



	__HYBRID__ UpdatePosition(	smartPtr<float> _gmass, smartPtr<float> _gposx, smartPtr<float> _gposy,smartPtr<float> _gposz,
								smartPtr<float> _gvelx, smartPtr<float> _gvely, smartPtr<float> _gvelz,
								smartPtr<float> _gaccx, smartPtr<float> _gaccy, smartPtr<float> _gaccz,
								unsigned long _lower, unsigned long _upper):
								gmass(_gmass),gposx(_gposx),gposy(_gposy),gposz(_gposz),
								gvelx(_gvelx),gvely(_gvely),gvelz(_gvelz),
								gaccx(_gaccx),gaccy(_gaccy),gaccz(_gaccz),
								lower(_lower),upper(_upper)
	{
		WORK_TYPE_ID = WORK_NBODY_UPDATEPOS | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~UpdatePosition(){}


	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		unsigned int range = (upper-lower);

//		unsigned int number_of_units = range / number;
//
//		if(number_of_units < 256) {
//				number = range / 256;
//				number_of_units = range / number;
//		}
		unsigned int number_of_units = 512;
		number = range / number_of_units;

		List<work*>* L = new List<work*>(number);

		unsigned start = lower;

		for (unsigned k = 0; k < number; k++) {

			UpdatePosition* up = (UpdatePosition*)LowLevelMemAllocator::sharedMemory(sizeof(UpdatePosition));
			if(up==NULL) {printf("ERRO!!!!!!!!!!!!!!!!1111\n");exit(0);}

			*up = UpdatePosition(gmass,gposx,gposy,gposz,gvelx,gvely,gvelz,gaccx,gaccy,gaccz,start,start+number_of_units);
			(*L)[k] = up;
			start+=number_of_units;

		}

		return L;
	}



	template<DEVICE_TYPE>
	__DEVICE__ void execute() {
		if(TID > (upper-lower)) return;
		unsigned long tid = TID + lower;

		float dvelx,dvely,dvelz;
		float velhx,velhy,velhz;

		for(; tid < upper; tid+=TID_SIZE) {
			dvelx = gaccx[tid] * dthf;
			dvely = gaccy[tid] * dthf;
			dvelz = gaccz[tid] * dthf;

			velhx = gvelx[tid] + dvelx;
			velhy = gvely[tid] + dvely;
			velhz = gvelz[tid] + dvelz;

			gposx[tid] += velhx * dtime;
			gposy[tid] += velhy * dtime;
			gposz[tid] += velhz * dtime;

			gvelx[tid] = velhx + dvelx;
			gvely[tid] = velhy + dvely;
			gvelz[tid] = velhz + dvelz;
		}

	}

	std::vector<pointerInfo>* toCacheR(){
		std::vector<pointerInfo>* L = new std::vector<pointerInfo>(6);
//		size_t st = (NNODES+1)*sizeof(float);
//		L->push_back(pointerInfo(gposx.ptr,st));
//		L->push_back(pointerInfo(gposy.ptr,st));
//		L->push_back(pointerInfo(gposz.ptr,st));
//		L->push_back(pointerInfo(gvelx.ptr,st));
//		L->push_back(pointerInfo(gvely.ptr,st));
//		L->push_back(pointerInfo(gvelz.ptr,st));
//		L->push_back(pointerInfo(gaccx.ptr,st));
//		L->push_back(pointerInfo(gaccy.ptr,st));
//		L->push_back(pointerInfo(gaccz.ptr,st));
//		L->push_back(pointerInfo(gmass.ptr,st));
		return L;
	}

	std::vector<pointerInfo>* toCacheW(){
		std::vector<pointerInfo>* L = new std::vector<pointerInfo>(0);
		return L;
	}
};

#endif /* UPDATEPOS_H_ */
