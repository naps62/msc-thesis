/***************************************************************************
 *   Copyright (C) 1998-2010 by authors (see AUTHORS.txt )                 *
 *                                                                         *
 *   This file is part of LuxRays.                                         *
 *                                                                         *
 *   LuxRays is free software; you can redistribute it and/or modify       *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   LuxRays is distributed in the hope that it will be useful,            *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 *   LuxRays website: http://www.luxrender.net                             *
 ***************************************************************************/

#if !defined(LUXRAYS_DISABLE_OPENCL) && !defined(WIN32) && !defined(__APPLE__)
#include <GL/glx.h>
#endif
#include <cstdio>
using std::sprintf;

#include "luxrays/core/intersectiondevice.h"
#include "core.h"
//#include "luxrays/core/context.h"

namespace luxrays {

//------------------------------------------------------------------------------
// DeviceDescription
//------------------------------------------------------------------------------

void DeviceDescription::FilterOne(std::vector<DeviceDescription *> &deviceDescriptions)
{
	int gpuIndex = -1;
	int cpuIndex = -1;
	for (size_t i = 0; i < deviceDescriptions.size(); ++i) {
		if ((cpuIndex == -1) && (deviceDescriptions[i]->GetType() &
			DEVICE_TYPE_NATIVE_THREAD))
			cpuIndex = (int)i;
		else if ((gpuIndex == -1) && (deviceDescriptions[i]->GetType() &
			DEVICE_TYPE_OPENCL_GPU)) {
			gpuIndex = (int)i;
			break;
		}
	}

	if (gpuIndex != -1) {
		std::vector<DeviceDescription *> selectedDev;
		selectedDev.push_back(deviceDescriptions[gpuIndex]);
		deviceDescriptions = selectedDev;
	} else if (gpuIndex != -1) {
		std::vector<DeviceDescription *> selectedDev;
		selectedDev.push_back(deviceDescriptions[cpuIndex]);
		deviceDescriptions = selectedDev;
	} else
		deviceDescriptions.clear();
}

void DeviceDescription::Filter(const DeviceType type,
	std::vector<DeviceDescription *> &deviceDescriptions)
{
	if (type == DEVICE_TYPE_ALL)
		return;
	size_t i = 0;
	while (i < deviceDescriptions.size()) {
		if ((deviceDescriptions[i]->GetType() & type) == 0) {
			// Remove the device from the list
			deviceDescriptions.erase(deviceDescriptions.begin() + i);
		} else
			++i;
	}
}

std::string DeviceDescription::GetDeviceType(const DeviceType type)
{
	switch (type) {
		case DEVICE_TYPE_ALL:
			return "ALL";
		case DEVICE_TYPE_NATIVE_THREAD:
			return "NATIVE_THREAD";
		case DEVICE_TYPE_OPENCL_ALL:
			return "OPENCL_ALL";
		case DEVICE_TYPE_OPENCL_DEFAULT:
			return "OPENCL_DEFAULT";
		case DEVICE_TYPE_OPENCL_CPU:
			return "OPENCL_CPU";
		case DEVICE_TYPE_OPENCL_GPU:
			return "OPENCL_GPU";
		case DEVICE_TYPE_OPENCL_UNKNOWN:
			return "OPENCL_UNKNOWN";
		case DEVICE_TYPE_VIRTUAL:
			return "VIRTUAL";
		default:
			return "UNKNOWN";
	}
}

//------------------------------------------------------------------------------
// Device
//------------------------------------------------------------------------------

Device::Device(/*const Context *context,*/ const DeviceType type, const size_t index) :
	/*deviceContext(context),*/ deviceType(type) {
	deviceIndex = index;
	started = false;
	usedMemory = 0;
}

Device::~Device() {
}

void Device::Start() {
	assert (!started);
	started = true;
}

void Device::Stop() {
	assert (started);
	started = false;
}

//------------------------------------------------------------------------------
// Native Device Description
//------------------------------------------------------------------------------

void NativeThreadDeviceDescription::AddDeviceDescs(std::vector<DeviceDescription *> &descriptions) {
	unsigned int count = boost::thread::hardware_concurrency();

	// Build the descriptions
	char buf[64];
	for (size_t i = 0; i < count; ++i) {
		sprintf(buf, "NativeThread-%03d", (int)i);
		std::string deviceName = std::string(buf);

		descriptions.push_back(new NativeThreadDeviceDescription(deviceName, i));
	}
}

//------------------------------------------------------------------------------
// OpenCL Device Description
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// IntersectionDevice
//------------------------------------------------------------------------------

IntersectionDevice::IntersectionDevice(/*const Context *context,*/
	const DeviceType type, const size_t index) :
	Device(/*context,*/ type, index), dataSet(NULL),
	dataParallelSupport(true) {
}

IntersectionDevice::~IntersectionDevice() {
}

void IntersectionDevice::SetDataSet(const DataSet *newDataSet) {
	assert (!started);
	assert ((newDataSet == NULL) || ((newDataSet != NULL) && newDataSet->IsPreprocessed()));

	dataSet = newDataSet;
}

void IntersectionDevice::Start() {
	assert (dataSet != NULL);

	Device::Start();

	statsStartTime = WallClockTime();
	statsTotalSerialRayCount = 0.0;
	statsTotalDataParallelRayCount = 0.0;
	statsDeviceIdleTime = 0.0;
	statsDeviceTotalTime = 0.0;
}

}
