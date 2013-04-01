#ifndef __MY_VTABLE_H_
#define __MY_VTABLE_H_

#include <config/vtable.h>


//TODO
//#error add function definitions here
//example:
// #include <saxpy.h>

/**
 * Virtual Tables
 */

#ifndef __CUDACC__
// CPU work table
// eg: (executeFN)&saxpy::execute<CPU_X86>
const executeFN WORK_CPU_TABLE[WORK_TOTAL] = {
	&work::execute<CPU_X86>,
};
#endif

// CPU dice table
// eg: (diceFN)&saxpy::dice<CPU_X86>
const diceFN DICE_CPU_TABLE[WORK_TOTAL] = {
	&work::dice<CPU_X86>,
};

// CPU toCacheR table
// eg: (toCacheRFN)&saxpy::toCacheR
const toCacheRFN TOCACHER_CPU_TABLE[WORK_TOTAL] = {
	&work::toCacheR,
};

// CPU toCacheW table
// eg: (toCacheWFN)&saxpy::toCacheW
const toCacheRFN TOCACHEW_CPU_TABLE[WORK_TOTAL] = {
	&work::toCacheR,
};

#endif // __MY_VTABLE_H_
