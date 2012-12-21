

#ifndef __VTABLE_X86_H_
#define __VTABLE_X86_H_


#include <config/work.cfg.h>
#include <gamalib/work/work.h>

/*
 *
 * Include work description files
 *
 */

#if (SAMPLE == 1)
#include <samples/SAXPY/saxpy.h>
#endif
#if (SAMPLE == 2)
#include <samples/FFT/bitTable.h>
#include <samples/FFT/bitKernel.h>
#include <samples/FFT/nthrootsKernel.h>
#include <samples/FFT/firstSetKernel.h>
#include <samples/FFT/middleSetKernel.h>
#include <samples/FFT/lastSetKernel.h>
#endif
#if (SAMPLE == 3)
#include <samples/MB-NBODY/ForceCalculation.h>
#include <samples/MB-NBODY/UpdatePosition.h>
#endif
#if (SAMPLE == 4 || SAMPLE == 5 )
#include <samples/BH/BHForce.h>
#endif







typedef work pWork;

typedef void (work::*executeFN)(void);
typedef List<work*>* (work::*diceFN)(unsigned int &number);
typedef std::vector<pointerInfo>* (work::*toCacheRFN)();
typedef std::vector<pointerInfo>* (work::*toCacheWFN)();
//typedef bool (work::*cacheFN)(CACHE* _cache);
//typedef bool (work::*evictFN)(void);
//typedef void (work::*destructorFN)(void);

#ifndef __CUDACC__
const executeFN WORK_CPU_TABLE[WORK_TOTAL] = {
		&work::execute<CPU_X86>,
#if (SAMPLE == 1)
		(executeFN)&saxpy::execute<CPU_X86>,
#endif
#if (SAMPLE == 2)
		(executeFN)&bitTable::execute<CPU_X86>,
		(executeFN)&bitKernel::execute<CPU_X86>,
		(executeFN)&nthrootsKernel::execute<CPU_X86>,
		(executeFN)&firstSetKernel::execute<CPU_X86>,
		(executeFN)&middleSetKernel::execute<CPU_X86>,
		(executeFN)&lastSetKernel::execute<CPU_X86>,
#endif
#if (SAMPLE == 3)
		(executeFN)&ForceCalculation::execute<CPU_X86>,
		(executeFN)&UpdatePosition::execute<CPU_X86>,
#endif
#if (SAMPLE == 4 || SAMPLE == 5 )
		(executeFN)&BHForce::execute<CPU_X86>
#endif
};
#endif

const diceFN DICE_CPU_TABLE[WORK_TOTAL] = {
		&work::dice<CPU_X86>,
#if (SAMPLE == 1)
		(diceFN)&saxpy::dice<CPU_X86>,
#endif
#if (SAMPLE == 2)
		(diceFN)&bitTable::dice<CPU_X86>,
		(diceFN)&bitKernel::dice<CPU_X86>,
		(diceFN)&nthrootsKernel::dice<CPU_X86>,
		(diceFN)&firstSetKernel::dice<CPU_X86>,
		(diceFN)&middleSetKernel::dice<CPU_X86>,
		(diceFN)&lastSetKernel::dice<CPU_X86>,
#endif
#if (SAMPLE == 3)
		(diceFN)&ForceCalculation::dice<CPU_X86>,
		(diceFN)&UpdatePosition::dice<CPU_X86>,
#endif
#if (SAMPLE == 4 || SAMPLE == 5)
		(diceFN)&BHForce::dice<CPU_X86>
#endif
};

const toCacheRFN TOCACHER_CPU_TABLE[WORK_TOTAL] = {
	&work::toCacheR,
#if (SAMPLE == 1)
	(toCacheRFN)&saxpy::toCacheR,
#endif
#if (SAMPLE == 2)
	(toCacheRFN)&bitTable::toCacheR,
	(toCacheRFN)&bitKernel::toCacheR,
	(toCacheRFN)&nthrootsKernel::toCacheR,
	(toCacheRFN)&firstSetKernel::toCacheR,
	(toCacheRFN)&middleSetKernel::toCacheR,
	(toCacheRFN)&lastSetKernel::toCacheR,
#endif
#if (SAMPLE == 3)
	(toCacheRFN)&ForceCalculation::toCacheR,
	(toCacheRFN)&UpdatePosition::toCacheR,
#endif
#if (SAMPLE == 4 || SAMPLE == 5)
	(toCacheRFN)&BHForce::toCacheR
#endif
};

const toCacheRFN TOCACHEW_CPU_TABLE[WORK_TOTAL] = {
	&work::toCacheR,
#if (SAMPLE == 1)
	(toCacheWFN)&saxpy::toCacheW,
#endif
#if (SAMPLE == 2)
	(toCacheWFN)&bitTable::toCacheW,
	(toCacheWFN)&bitKernel::toCacheW,
	(toCacheWFN)&nthrootsKernel::toCacheW,
	(toCacheWFN)&firstSetKernel::toCacheW,
	(toCacheWFN)&middleSetKernel::toCacheW,
	(toCacheWFN)&lastSetKernel::toCacheW,
#endif
#if (SAMPLE == 3)
	(toCacheWFN)&ForceCalculation::toCacheW,
	(toCacheWFN)&UpdatePosition::toCacheW,
#endif
#if (SAMPLE == 4 || SAMPLE == 5)
	(toCacheWFN)&BHForce::toCacheW
#endif
};


#endif
