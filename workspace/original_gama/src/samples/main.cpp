/*
 * main.cpp
 *
 *  Created on: Apr 2, 2012
 *      Author: jbarbosa
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <math.h>

#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <errno.h>
#include <float.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <cublas.h>

#include <config/common.h>
#include <gamalib/gamalib.h>

MemorySystem* LowLevelMemAllocator::_memSys = NULL;

#if (SAMPLE == 0)
#include <samples/empty_work/empty-main.h>
#endif 
#if (SAMPLE == 1)
#include <samples/SAXPY/saxpy-main.h>
#endif 
#if (SAMPLE == 2)
#include <samples/FFT/fft-main.h>
#endif 
#if (SAMPLE == 3)
#include <samples/MB-NBODY/mb-nbody-main.h>
#endif 
#if (SAMPLE == 4)
#include <samples/BH/bh-main.h>
#endif 
#if (SAMPLE == 5)
#include <samples/BH2/bh2-main.h>
#endif
