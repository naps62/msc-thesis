/*
 * devices.h
 *
 *  Created on: Dec 14, 2012
 *      Author: Miguel Palhas
 */

#ifndef __MY_DEVICES_H_
#define __MY_DEVICES_H_

#include <config/system.cfg.h>

/**
 * Device list
 * Details about each device that GAMA should use
 * Originally from: config/system.cfg.h
 */

#define CPU_SINGLE
//#define CPU_ONLY

#if defined(CPU_SINGLE)

const unsigned TOTAL_DEVICES = 1;
const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES]      = { CPU_X86 };
const unsigned    TOTAL_CORES_TYPES[TOTAL_DEVICES] = { 1,      };
const float       LOADS[TOTAL_DEVICES]             = { 1.0f    };

#elif defined(CPU_ONLY)

const unsigned TOTAL_DEVICES = 4;
const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES]      = { CPU_X86, CPU_X86, CPU_X86, CPU_X86 };
const unsigned    TOTAL_CORES_TYPES[TOTAL_DEVICES] = { 1,       1,       1,       1,      };
const float       LOADS[TOTAL_DEVICES]             = { 0.25f,   0.25f,   0.25f,   0.25f,  };

#else

// Total number of devices to be used, regardless of type
//#error Did not check device list
const unsigned TOTAL_DEVICES = 5;
const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES]      = { CPU_X86, CPU_X86, CPU_X86, CPU_X86, GPU_CUDA };
const unsigned    TOTAL_CORES_TYPES[TOTAL_DEVICES] = { 1,       1,       1,       1,       16       };
const float       LOADS[TOTAL_DEVICES]             = { 0.05f,   0.05f,   0.05f,   0.05f,   .8f      };

#endif

#endif // __MY_DEVICES_H_
