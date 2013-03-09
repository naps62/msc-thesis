/*
 * CudaAllocator.h
 *
 *  Created on: Apr 2, 2012
 *      Author: jbarbosa
 */

#ifndef MEMALLOCATOR_H_
#define MEMALLOCATOR_H_

#include "LowLevelMemAllocator.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <stdlib.h>

#include <assert.h>
#include "MemorySystem.h"

class LowLevelMemAllocator {

public:

	static MemorySystem *_memSys;

	static void* deviceMemory(size_t size) {
			void* ptr;
			(cudaMalloc((void**)&ptr, size));
			return ptr;
	}
	static void* hostMemory(size_t size) {
		void* ptr;
		if(posix_memalign(&ptr,4096,size)) {
			return NULL;
		}
		return ptr;
	}
	__DEVICE__
	static void* sharedMemory(size_t size) {
		void* ptr;
		//cutilDrvSafeCall(cuMemHostAlloc((void**)&ptr, size,CU_MEMHOSTALLOC_PORTABLE));
		ptr = ((MemorySystem*)_gmem)->allocate(size);
		return ptr;
	}

	static void freeDevice(void* ptr) {
		assert(ptr!=NULL);
		(cudaFree(ptr));
	}

	__DEVICE__
	static void freeHost(void* ptr) {
		assert(ptr!=NULL);
		((MemorySystem*)_gmem)->deallocate(ptr);
		//free(ptr);
	}
	__DEVICE__
	static void freeShared(void* ptr) {
		assert(ptr!=NULL);
		((MemorySystem*)_gmem)->deallocate(ptr);
	}

	__DEVICE__
	static void* alloc(size_t size, MEM_TYPE MT) {
		switch(MT) {
			case HOST: return LowLevelMemAllocator::hostMemory(size);
			case SHARED: return LowLevelMemAllocator::sharedMemory(size);
			case DEVICE: return LowLevelMemAllocator::deviceMemory(size);
		}
		return NULL;
	}

	__DEVICE__
	static void dealloc(void* ptr, MEM_TYPE MT) {
		switch(MT) {
			case HOST: LowLevelMemAllocator::freeHost(ptr); break;
			case SHARED: LowLevelMemAllocator::freeShared(ptr); break;
			case DEVICE: LowLevelMemAllocator::freeDevice(ptr); break;
		}
	}

	__DEVICE__
	static void dealloc(void* ptr) {
		if((unsigned long)ptr >= (unsigned long)((MemorySystem*)_gmem)->MEM_POOL && (unsigned long)ptr <= ((unsigned long)((MemorySystem*)_gmem)->MEM_POOL + MEM_SIZE)) {
			LowLevelMemAllocator::freeShared(ptr);
		} else {
			free(ptr);
		}
	}
};

#ifndef MEMSYS_M
#define MEMSYS_M
	//MemorySystem* LowLevelMemAllocator::_memSys = NULL;
#endif // MEMSYSY_M
#endif /* MEMALLOCATOR_H_ */
