/*
 * common.h
 *
 *  Created on: Apr 26, 2011
 *      Author: jbarbosa
 */

#ifndef COMMON_H_
#define COMMON_H_

#ifndef SAMPLE
#define SAMPLE 1
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "global.def.h"
#include "macros.def.h"
#include "work.cfg.h"
// ***** Cache ******
 #define GAMA_CACHE
// ******************
#include "memory.cfg.h"
#include "system.cfg.h"
#include "scheduler.cfg.h"
#include "workqueue.cfg.h"
#include "problem.cfg.h"

#include "pthread_barrier.h"


// #ifndef SAMPLE
// #define SAMPLE 3
// #endif


#endif /* COMMON_H_ */