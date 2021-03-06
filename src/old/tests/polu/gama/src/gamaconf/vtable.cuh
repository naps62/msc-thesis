/*
 * vtable.cuh
 *
 *  Created on: Dec 14, 2012
 *      Author: naps62
 */

#ifndef __MY_VTABLE_CUH_
#define __MY_VTABLE_CUH_

#include <config/vtable.cuh>

// GPU work table
// eg: (executeFN)&saxpy::execute<GPU_CUDA>
__constant__ executeFN WORK_GPU_TABLE[WORK_TOTAL] = {
	&work::execute<GPU_CUDA>,
	(executeFN)&ComputeFlux::execute<GPU_CUDA>,
	(executeFN)&Update::execute<GPU_CUDA>
};

#endif // __MY_VTABLE_CUH_
