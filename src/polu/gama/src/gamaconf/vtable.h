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
// #include <saxpy.h>

/**
 * Virtual Tables
 */
#define __GAMA_SKEL
#ifndef __CUDACC__
// CPU work table
// eg: (executeFN)&saxpy::execute<CPU_X86>
const executeFN WORK_CPU_TABLE[WORK_TOTAL] = {
	&work::execute<CPU_X86>,
	#ifndef __GAMA_SKEL
		#error add cpu work table
	#endif
};
#endif

// CPU dice table
// eg: (diceFN)&saxpy::dice<CPU_X86>
const diceFN DICE_CPU_TABLE[WORK_TOTAL] = {
	&work::dice<CPU_X86>,
	#ifndef __GAMA_SKEL
		#error add cpu dice table
	#endif
};

// CPU toCacheR table
// eg: (toCacheRFN)&saxpy::toCacheR
const toCacheRFN TOCACHER_CPU_TABLE[WORK_TOTAL] = {
	&work::toCacheR,
	#ifndef __GAMA_SKEL
		#error add cpu toCacheR cpu table
	#endif
};

// CPU toCacheW table
// eg: (toCacheWFN)&saxpy::toCacheW
const toCacheRFN TOCACHEW_CPU_TABLE[WORK_TOTAL] = {
	&work::toCacheR,
	#ifndef __GAMA_SKEL
		#error add toCacheW cpu table
	#endif
};

#endif // __MY_VTABLE_H_
