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
const unsigned TOTAL_DEVICES = 1;
const DEVICE_TYPE DEVICE_TYPES[TOTAL_DEVICES]      = { GPU_CUDA };	// Device type arrangement (eg: CPU_X86, GPU_CUDA)
const unsigned    TOTAL_CORES_TYPES[TOTAL_DEVICES] = { 16       };	// cores available per device (1 for CPU_X86. In GPU_CUDA, should be the number of SM's)
const float       LOADS[TOTAL_DEVICES]             = { 1.f      };	// Static work load percentage per device

#endif // __MY_DEVICES_H_
