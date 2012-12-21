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

// Total number of devices to be used, regardless of type
#error Did not check device list
const unsigned TOTAL_DEVICES = 5;
// Device type arrangement (eg: CPU_X86, GPU_CUDA)
const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES]      = { CPU_X86, CPU_X86, CPU_X86, CPU_X86, GPU_CUDA };
// Total number of cores available per device (tipically 1 for CPU_X86. In GPU_CUDA, should be the number of SM's)
const unsigned    TOTAL_CORES_TYPES[TOTAL_DEVICES] = { 1,       1,       1,       1,       16       };
// Static work load percentage per device
const float       LOADS[TOTAL_DEVICES]             = { 0.05f,   0.05f,   0.05f,   0.05f,   .8f      };

#endif // __MY_DEVICES_H_
