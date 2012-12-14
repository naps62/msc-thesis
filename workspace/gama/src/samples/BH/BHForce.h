/*
 * BHForce.h
 *
 *  Created on: Aug 7, 2012
 *      Author: ricardo
 */

#ifndef BHFORCE_H_
#define BHFORCE_H_

class BHForce : public work {
public:

	smartPtr<float> mass;
	smartPtr<float> posx;
	smartPtr<float> posy;
	smartPtr<float> posz;
	smartPtr<float> velx;
	smartPtr<float> vely;
	smartPtr<float> velz;
	smartPtr<float> accx;
	smartPtr<float> accy;
	smartPtr<float> accz;

	smartPtr<int> child;
	smartPtr<int> sort;

	unsigned long nnodes;
	int maxdepth;
	float radius;

	unsigned long lower;
	unsigned long upper;
	unsigned long step;

	float *dq;

	unsigned int* count;

	__HYBRID__ BHForce(	smartPtr<float> _gmass, smartPtr<float> _gposx, smartPtr<float> _gposy,smartPtr<float> _gposz,
						smartPtr<float> _gvelx, smartPtr<float> _gvely, smartPtr<float> _gvelz,
						smartPtr<float> _gaccx, smartPtr<float> _gaccy, smartPtr<float> _gaccz,
						smartPtr<int> _gchild, smartPtr<int> _gsort,
						unsigned long _lower, unsigned long _upper, unsigned long _step,float* _dq,unsigned long _nnodesd, int maxd, float grad,unsigned int* c):

						mass(_gmass),posx(_gposx),posy(_gposy),posz(_gposz),
						velx(_gvelx),vely(_gvely),velz(_gvelz),
						accx(_gaccx),accy(_gaccy),accz(_gaccz),
						child(_gchild),sort(_gsort),
						lower(_lower),upper(_upper), step(_step), dq(_dq), nnodes(_nnodesd), maxdepth(maxd), radius(grad), count(c)
		{
			WORK_TYPE_ID = WORK_BH_FORCE | W_REGULAR | W_WIDE;
		}

	__HYBRID__ ~BHForce(){}


	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {
		unsigned int range = (upper-lower);

		unsigned int number_of_units = (range / number);

		if(number_of_units == 0) {
			number_of_units = 1;
			number = range;
		}

		unsigned int size = (range % number) ? number + 1 : number;

		List<work*>* L = new List<work*>(size);

		unsigned start = lower;

		for (unsigned k = 0; k < number; k++) {

			BHForce* fc = (BHForce*)LowLevelMemAllocator::sharedMemory(sizeof(BHForce));
			if(fc==NULL) {printf("ERRO!!!!!!!!!!!!!!!!1111\n");exit(0);}

			*fc = BHForce(mass,posx,posy,posz,velx,vely,velz,accx,accy,accz,child,sort,start,start+number_of_units,step,dq,nnodes,maxdepth,radius,count);
			(*L)[k] = fc;
			start+=number_of_units;

		}

		if(range % number) {
			BHForce* fc = (BHForce*)LowLevelMemAllocator::sharedMemory(sizeof(BHForce));
			if(fc==NULL) {printf("ERRO!!!!!!!!!!!!!!!!1111\n");exit(0);}
			*fc = BHForce(mass,posx,posy,posz,velx,vely,velz,accx,accy,accz,child,sort,start,upper,step,dq,maxdepth,nnodes,radius,count);
			(*L)[size-1] = fc;
		}

		return L;
	}

	template<DEVICE_TYPE>
	__DEVICE__ void execute() {

	}



	std::vector<pointerInfo>* toCacheR(){
		std::vector<pointerInfo>* L = new std::vector<pointerInfo>(6);
		size_t st = (NNODES+1)*sizeof(float);
		L->push_back(pointerInfo(posx.ptr,st));
		L->push_back(pointerInfo(posy.ptr,st));
		L->push_back(pointerInfo(posz.ptr,st));
		L->push_back(pointerInfo(mass.ptr,st));
		//L->push_back(pointerInfo(child.ptr,((NNODES+1)*8)*sizeof(int)));
		L->push_back(pointerInfo(sort.ptr,st));
		return L;
	}

	std::vector<pointerInfo>* toCacheW(){
		std::vector<pointerInfo>* L = new std::vector<pointerInfo>(0);
		return L;
	}



};

#ifdef __CUDACC__
template<> __DEVICE__ void BHForce::execute<GPU_CUDA>();
#else
template <>
void __DEVICE__ BHForce::execute<CPU_X86>();
#endif

#endif /* BHFORCE_H_ */
