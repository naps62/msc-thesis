/*
 * gemm.h
 *
 *  Created on: May 22, 2012
 *      Author: amariano
 */

#ifndef SYNTHA_H_
#define SYNTHA_H_

#ifndef TYPE
#define TYPE float
#endif

#include <gamalib/memlib/MemoryPool.h>
#include "Particle.h"

class syntha : public work {

public:

	unsigned int seed;

	__HYBRID__ syntha(){
	}

	__HYBRID__ syntha(unsigned _i) : seed(_i)
	{
		WORK_TYPE_ID = WORK_SYNTHA;
	}

	__HYBRID__ ~syntha() {
	}


	template<DEVICE_TYPE>
	__DEVICE__ List<work*>* dice(unsigned int &number) {

		List<work*>* L = new List<work*>(number);

		MemoryPool<syntha> pool(number);

		for (unsigned i = 0; i < number; i++) {

			syntha* s = pool.pool();
			*s = syntha(i);

			(*L)[i] = s;

		}

		return L;
	}

	template<DEVICE_TYPE>
	__HYBRID__ void execute() {

		//Particle* p = new Particle(0);
		//p->process();

	}


};

#endif /* INITARRAY_H_ */
