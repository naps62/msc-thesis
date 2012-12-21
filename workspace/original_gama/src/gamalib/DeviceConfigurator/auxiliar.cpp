/*
 * auxiliar.cpp
 *
 *  Created on: May 3, 2012
 *      Author: jbarbosa
 */




#include <cuda.h>
#include <helper_cuda.h>


#include <config/common.h>


#include "DeviceCuda.h"

#include <gamalib/utils/cuda_utils.cuh>

#include <gamalib/utils/x86_utils.h>


void del(Workqueue<work, INBOX_QUEUE_SIZE, CPU_X86>* INBOX) {
	for(int i=0; i < INBOX_QUEUE_SIZE; i++)
	if(INBOX->data[i] != NULL) {
		delete INBOX->data[i];
	}
	delete INBOX;
}
