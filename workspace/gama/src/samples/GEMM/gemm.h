/*
 * gemm.h
 *
 *  Created on: May 22, 2012
 *      Author: amariano
 */

#ifndef GEMM_H_
#define GEMM_H_

#ifndef TYPE
#define TYPE float
#endif

#include <gamalib/memlib/MemoryPool.h>

class gemm : public work {

public:
	smartPtr<float> A;
	smartPtr<float> B;
	smartPtr<float> C;

	unsigned int N;
	unsigned int pos;
	unsigned int i;
	unsigned int endJ;
	unsigned int endK;

	__HYBRID__ gemm(){
	}

	__HYBRID__ gemm(smartPtr<float> _A, smartPtr<float> _B, smartPtr<float> _C, unsigned _N,  unsigned _pos, unsigned _i) : A(_A), B(_B), C(_C), endJ(_N), endK(_N), N(_N), pos(_pos), i(_i)
	{
		WORK_TYPE_ID = WORK_GEMM | WD_NOT_DICEABLE | W_REGULAR | W_WIDE;
	}

	__HYBRID__ ~gemm() {
	}


	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

//		unsigned int work_units = N / number;
//
//		if(work_units == 0) {
//			work_units = 1;
//			//number = range;
//		}

		//List<work*>* L = new List<work*>(number);
		List<work*>* L = new List<work*>(N*N);

		for (unsigned i = 0; i < N; i++) {

			for (unsigned k = 0; k < N; k++) {

				gemm* s = (gemm*)LowLevelMemAllocator::sharedMemory(sizeof(gemm));
				*s = gemm(A,B,C,N,k+i*N,i);

				(*L)[i*N+k] = s;

			}

		}

		return L;
	}

	template<DEVICE_TYPE>
	__HYBRID__ void execute() {

		i = pos/N;
		int j = pos-((pos/N)*N);

		//printf("Vou escrever na posição [%d][%d] e estou com i = %d, j= %d!\n",pos/N,pos-((pos/N)*N),i,j);

		float sum = 0;

		for(int k = 0; k < N; k++){
			sum += A[i*N+k] * B[k*N+j];
		}

		C[pos] = sum;
	}


};

#endif /* INITARRAY_H_ */
