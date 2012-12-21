#ifndef __VTABLE_CUDA_H_
#define __VTABLE_CUDA_H_

#include <gamalib/work/work.h>
#include <config/vtable.h>

__constant__ executeFN WORK_GPU_TABLE[WORK_TOTAL] = {

		&work::execute<CPU_X86>,
#if (SAMPLE == 1)
		(executeFN)&saxpy::execute<GPU_CUDA>,
#endif
#if (SAMPLE == 2)
		(executeFN)&bitTable::execute<GPU_CUDA>,
		(executeFN)&bitKernel::execute<GPU_CUDA>,
		(executeFN)&nthrootsKernel::execute<GPU_CUDA>,
		(executeFN)&firstSetKernel::execute<GPU_CUDA>,
		(executeFN)&middleSetKernel::execute<GPU_CUDA>,
		(executeFN)&lastSetKernel::execute<GPU_CUDA>,
#endif
#if (SAMPLE == 3)
		(executeFN)&ForceCalculation::execute<GPU_CUDA>,
		(executeFN)&UpdatePosition::execute<GPU_CUDA>,
#endif
#if (SAMPLE == 4 || SAMPLE == 5 )
		(executeFN)&BHForce::execute<GPU_CUDA>
#endif
};



//__constant__ diceFN DICE_GPU_TABLE[WORK_TOTAL] = {
//
//		&work::dice<GPU_CUDA>,
//		(diceFN)&saxpy::dice<GPU_CUDA>
//
//};



#endif
