/*
 * ForceCalculation.h
 *
 *  Created on: Jul 24, 2012
 *      Author: ricardo
 */

#ifndef FORCECALCULATION_H_
#define FORCECALCULATION_H_


class ForceCalculation : public work {
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



	__HYBRID__ ForceCalculation(smartPtr<float> _gmass, smartPtr<float> _gposx, smartPtr<float> _gposy,smartPtr<float> _gposz,
								smartPtr<float> _gvelx, smartPtr<float> _gvely, smartPtr<float> _gvelz,
								smartPtr<float> _gaccx, smartPtr<float> _gaccy, smartPtr<float> _gaccz,
								unsigned long _lower, unsigned long _upper, unsigned long _step):
								gmass(_gmass),gposx(_gposx),gposy(_gposy),gposz(_gposz),
								gvelx(_gvelx),gvely(_gvely),gvelz(_gvelz),
								gaccx(_gaccx),gaccy(_gaccy),gaccz(_gaccz),
								lower(_lower),upper(_upper), step(_step)
	{
		WORK_TYPE_ID = WORK_NBODY_FORCE | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~ForceCalculation(){}


	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		unsigned int range = (upper-lower);

		unsigned int number_of_units = 512;
		number = range / number_of_units;

		List<work*>* L = new List<work*>(number);

		unsigned start = lower;

		for (unsigned k = 0; k < number; k++) {

			ForceCalculation* fc = (ForceCalculation*)LowLevelMemAllocator::sharedMemory(sizeof(ForceCalculation));
			if(fc==NULL) {printf("ERRO!!!!!!!!!!!!!!!!1111\n");exit(0);}

			*fc = ForceCalculation(gmass,gposx,gposy,gposz,gvelx,gvely,gvelz,gaccx,gaccy,gaccz,start,start+number_of_units,step);
			(*L)[k] = fc;
			start+=number_of_units;

		}

		return L;
	}



	template<DEVICE_TYPE>
	__DEVICE__ void execute() {
		if(TID > (upper-lower)) return;
		unsigned long tid = TID + lower;

		for(; tid < upper; tid+=TID_SIZE) {


			float ax=.0f,ay=.0f,az=.0f;
			float px,py,pz;

			px = gposx[tid];
			py = gposy[tid];
			pz = gposz[tid];

			for(int n=0; n<NBODIES; n++){
				float dx,dy,dz,tmp;

				dx = gposx[n] - px;
				dy = gposy[n] - py;
				dz = gposz[n] - pz;

				tmp = dx*dx + (dy*dy + (dz*dz + epssq));  // compute distance squared (plus softening)
#ifdef __CUDACC__
				tmp = rsqrtf(tmp);
#else
				tmp = 1.0f/sqrtf(tmp); // compute distance
#endif
				tmp = gmass[n] * tmp * tmp * tmp;
				ax += dx * tmp;
				ay += dy * tmp;
				az += dz * tmp;

			}

			if (step > 0) {
				gvelx[tid] += (ax - gaccx[tid]) * dthf;
				gvely[tid] += (ay - gaccy[tid]) * dthf;
				gvelz[tid] += (az - gaccz[tid]) * dthf;
			}

			gaccx[tid] = ax;
			gaccy[tid] = ay;
			gaccz[tid] = az;

		}
	}


	std::vector<pointerInfo>* toCacheR(){
		std::vector<pointerInfo>* L = new std::vector<pointerInfo>(6);
		size_t st = (NNODES+1)*sizeof(float);
		L->push_back(pointerInfo(gposx.ptr,st));
		L->push_back(pointerInfo(gposy.ptr,st));
		L->push_back(pointerInfo(gposz.ptr,st));
		L->push_back(pointerInfo(gmass.ptr,st));
		return L;
	}

	std::vector<pointerInfo>* toCacheW(){
		std::vector<pointerInfo>* L = new std::vector<pointerInfo>(0);
		return L;
	}


};

#endif /* FORCECALCULATION_H_ */
