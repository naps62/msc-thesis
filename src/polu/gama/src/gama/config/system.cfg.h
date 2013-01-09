/*
 * system.h
 *
 *  Created on: Apr 3, 2012
 *      Author: jbarbosa
 */

#ifndef SYSTEM_H_
#define SYSTEM_H_

enum DEVICE_TYPE {
	CPU_X86 = 0,	// X86 compatible
	GPU_CUDA	// GPU_CUDA
};

enum MEM_TYPE {
	HOST = 0,
	SHARED,
	DEVICE
};

//#ifndef SCONF
//#define SCONF 4
//#endif


const unsigned long NTHREAD = 368;

#include <gamaconf/devices.h>

//#if (SCONF==0)
//const unsigned TOTAL_DEVICES = 1;
//const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES] = {GPU_CUDA};
//const unsigned TOTAL_CORES_TYPES[TOTAL_DEVICES] = {16};
//const float LOADS[TOTAL_DEVICES] = {1.f};
//#endif
//
//#if (SCONF==1)
//const unsigned TOTAL_DEVICES = 3;
//const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES] = {CPU_X86,CPU_X86,GPU_CUDA};
//const unsigned TOTAL_CORES_TYPES[TOTAL_DEVICES] = {1,16,16};
//const float LOADS[TOTAL_DEVICES] = {0.25f,0.25f,0.5f};
//#endif
//
//#if (SCONF==2)
//const unsigned TOTAL_DEVICES = 4;
//const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES] = {CPU_X86,CPU_X86,CPU_X86,CPU_X86};
//const unsigned TOTAL_CORES_TYPES[TOTAL_DEVICES] = {1,1,1,1};
//const float LOADS[TOTAL_DEVICES] = {0.25f,0.25f,0.25f,0.25f};
//#endif
//
//#if (SCONF==4)
//const unsigned TOTAL_DEVICES = 5;
//const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES] = {CPU_X86,CPU_X86,CPU_X86,CPU_X86,GPU_CUDA};
//const unsigned TOTAL_CORES_TYPES[TOTAL_DEVICES] = {1,1,1,1,16};
//const float LOADS[TOTAL_DEVICES] = {0.05f,0.05f,0.05f,0.05f,.8f};
//#endif
//
//#if (SCONF==5)
//const unsigned TOTAL_DEVICES = 7;
//const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES] = {CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,GPU_CUDA};
//const unsigned TOTAL_CORES_TYPES[TOTAL_DEVICES] = {1,1,1,1,1,1,3};
//const float LOADS[TOTAL_DEVICES] = {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.4f};
//#endif
//
//#if (SCONF==6)
//const unsigned TOTAL_DEVICES = 8;
//const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES] = {CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86};
//const unsigned TOTAL_CORES_TYPES[TOTAL_DEVICES] = {1,1,1,1};
//const float LOADS[TOTAL_DEVICES] = {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.3f};
//#endif
//
//#if (SCONF==7)
//const unsigned TOTAL_DEVICES = 9;
//const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES] = {CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,GPU_CUDA};
//const unsigned TOTAL_CORES_TYPES[TOTAL_DEVICES] = {1,1,1,1,1,1,1,1,3};
//const float LOADS[TOTAL_DEVICES] = {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.2f};
//#endif
//
//#if (SCONF==8)
//const unsigned TOTAL_DEVICES = 16+2;
//const float P = 1.f / (float) TOTAL_DEVICES;
//const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES] = {
//    CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,
//    CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,CPU_X86,
//    GPU_CUDA,GPU_CUDA};
//const unsigned TOTAL_CORES_TYPES[TOTAL_DEVICES] = {
//    1,1,1,1,1,1,1,1,
//    1,1,1,1,1,1,1,1,
//    16,16};
//const float LOADS[TOTAL_DEVICES] = {
//    P, P, P, P, P, P, P, P,
//    P, P, P, P, P, P, P, P,
//    P, P
//};
//#endif



#endif /* SYSTEM_H_ */
