/*
 * vtable.h
 *
 *  Created on: Dec 14, 2012
 *     Author: Miguel Palhas
 */

#ifndef __MY_VTABLE_H_
#define __MY_VTABLE_H_

#include <config/vtable.h>


//TODO
//#error add function definitions here
//example:
#include <saxpy.h>

/**
 * Virtual Tables
 */

#ifndef __CUDACC__
// CPU work table
// eg: (executeFN)&saxpy::execute<CPU_X86>
const executeFN WORK_CPU_TABLE[WORK_TOTAL] = {
	&work::execute<CPU_X86>,
	//#error add cpu work table
	(executeFN)&saxpy::execute<CPU_X86>
};
#endif

// CPU dice table
// eg: (diceFN)&saxpy::dice<CPU_X86>
const diceFN DICE_CPU_TABLE[WORK_TOTAL] = {
	&work::dice<CPU_X86>,
	//#error add cpu dice table
	(diceFN)&saxpy::dice<CPU_X86>
};

// CPU toCacheR table
// eg: (toCacheRFN)&saxpy::toCacheR
const toCacheRFN TOCACHER_CPU_TABLE[WORK_TOTAL] = {
	&work::toCacheR,
	//#error add cpu toCacheR cpu table
	//(toCacheRFN)&saxpy::toCacheR
};

// CPU toCacheW table
// eg: (toCacheWFN)&saxpy::toCacheW
const toCacheRFN TOCACHEW_CPU_TABLE[WORK_TOTAL] = {
	&work::toCacheR,
	//#error add toCacheW cpu table
	//(toCacheWFN)&saxpy::toCacheW
};

#endif // __MY_VTABLE_H_
