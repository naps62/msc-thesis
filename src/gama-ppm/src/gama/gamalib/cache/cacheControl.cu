/*
 * cacheControl.cpp
 *
 *  Created on: Sep 26, 2012
 *      Author: jbarbosa
 */
#include <cuda.h>
#include <driver_types.h>
#include <helper_cuda.h>

#include <config/common.h>

#include <vector>

#include "cacheControl.h"

#include <gamalib/utils/x86_utils.h>

#include "CacheList.h"

struct cacheCopy {
	unsigned long start;
	unsigned int pages;
};

template<> cacheControl<GPU_CUDA>::cacheControl() :
		cache(0), pages_cache(0), pages_local(0), pages_inflight(0), _lock(0), _lock_all(
				0), _copies_in_flight(0), total_time(0.f) {

	//get the amount of free memory on the graphics card
	size_t free;
	size_t total;

	checkCudaErrors(cudaMemGetInfo(&free,&total));

//	printf("GPU Cache %u %u (%u)\n", free / 1024 / 1024,
//			CACHE_SIZE / 1024 / 1024, total / 1024 / 1024);

	if (free < CACHE_SIZE) {
		printf("Out of GPU memory\n");
		exit(0);
	}

	checkCudaErrors(cudaMalloc((void**)&cache,CACHE_SIZE));
	checkCudaErrors(cudaMemcpyToSymbol(_cache,&cache,sizeof(void*)));
	checkCudaErrors(cudaMalloc((void**)&pages_cache,NUMBER_PAGES*sizeof(unsigned long)));
	checkCudaErrors(cudaMemcpyToSymbol(_pages,&pages_cache,sizeof(void*)));
	checkCudaErrors(cudaMemset(pages_cache,0,NUMBER_PAGES*sizeof(unsigned long)));

	unsigned int a = posix_memalign((void**) &pages_local, 4096,
			NUMBER_PAGES * sizeof(unsigned long));
	a = posix_memalign((void**) &pages_inflight, 4096,
			NUMBER_PAGES * sizeof(unsigned long));
	a = posix_memalign((void**) &_lock, 4096,
			NUMBER_PAGES * sizeof(unsigned int));

}

template<> cacheControl<GPU_CUDA>::~cacheControl() {
	checkCudaErrors(cudaFree(cache));
	checkCudaErrors(cudaFree(pages_cache));
	cached.clear();
	free(pages_local);
	free(pages_inflight);

}

template<> void cacheControl<GPU_CUDA>::cacheReset() {
	memset(pages_local, 0, NUMBER_PAGES * sizeof(unsigned long));
	checkCudaErrors(cudaMemset(pages_cache,0,NUMBER_PAGES*sizeof(unsigned long)));
	cached.clear();
}

template<> void cacheControl<GPU_CUDA>::cacheMove(unsigned long addr_start,
		unsigned long size, cudaStream_t* st) {
	void* addr_c =
			(void*) ((unsigned long) cache + (addr_start & offset_filter));

	if ((unsigned long) addr_c + size < (unsigned long) cache + CACHE_SIZE) {

		checkCudaErrors(cudaMemcpyAsync( addr_c, // dst
				(void*)addr_start,// src
				size,// size
				cudaMemcpyHostToDevice, *st// stream
				));
	} else {
		unsigned int left = ((unsigned long) addr_c + size)
				- ((unsigned long) cache + CACHE_SIZE);
		unsigned int copy = size - left;
		checkCudaErrors(cudaMemcpyAsync( addr_c, // dst
				(void*)addr_start,// src
				copy,// size
				cudaMemcpyHostToDevice, *st// stream
				));

		if (left > CACHE_SIZE - copy) {
			printf("Error in cache");
			exit(0);
		}
		checkCudaErrors(cudaMemcpyAsync( addr_c, // dst
				(void*)addr_start,// src
				left,// size
				cudaMemcpyHostToDevice, *st// stream
				));
	}
}

template<> void cacheControl<GPU_CUDA>::pageLoad(unsigned long idx,
		unsigned long size, cudaStream_t* st) {
	unsigned int max = CACHE_SIZE >> BLOCK_BITS;
	if (idx + size < max) {
		checkCudaErrors(cudaMemcpyAsync( &pages_cache[idx], // dst
				&pages_local[idx],// src
				sizeof(unsigned long) * size,// size
				cudaMemcpyHostToDevice, *st// stream
				));
	} else {
		unsigned int left = ((idx + size) - max);

		unsigned int copy = size - left;
		checkCudaErrors(cudaMemcpyAsync( &pages_cache[idx], // dst
				&pages_local[idx],// src
				sizeof(unsigned long) * copy,// size
				cudaMemcpyHostToDevice, *st// stream
				));

		if (left > max - copy) {
			printf("Error in cache page load...\n");
			exit(0);
		}
		checkCudaErrors(cudaMemcpyAsync( &pages_cache[0], // dst
				&pages_local[0],// src
				sizeof(unsigned long) * left,// size
				cudaMemcpyHostToDevice, *st// stream
				));

	}
}

template<> void cacheControl<GPU_CUDA>::emptyLocked(
		std::vector<unsigned long> &locked) {
	unsigned long idx;
	while (!locked.empty()) {
		idx = locked.back();
		locked.pop_back();
		__sync_val_compare_and_swap((volatile unsigned int*) &_lock[idx], 1, 0);
	}
};


template<> void cacheControl<GPU_CUDA>::cachePtr(void* ptr, size_t size, cudaStream_t* st, CACHE_TYPE CT) {

//    unsigned long addr_start = ((unsigned long)ptr &address_filter);
    unsigned long size_page = (size % BLOCK_SIZE) ? (size / BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE : size;
    unsigned long addr_end = ((unsigned long) ptr + size_page);

    for (unsigned long addr_curr = (unsigned long) ptr & address_filter; addr_curr < addr_end; addr_curr += BLOCK_SIZE) {
		unsigned long idx = (addr_curr & index_filter) >> BLOCK_BITS;
		unsigned long value = __sync_val_compare_and_swap(
				(volatile unsigned long*) &pages_local[idx], 0, addr_curr);
		if (value == addr_curr || value == 0) {
			__sync_fetch_and_add((volatile unsigned long*) &pages_inflight[idx],
					1);
		}
		if (value == 0) {
			cl.insert(Range(addr_curr, addr_curr + BLOCK_SIZE));
		}
	}
	
    Range r;

	while (!(r = cl.next()).isInvalid()) {
		cacheMove(r.addr_start, r.addr_end - r.addr_start, st);
        cached.insert(r);
		unsigned long idx = (r.addr_start & index_filter) >> BLOCK_BITS;
		unsigned size = (r.addr_end - r.addr_start) >> BLOCK_BITS;
		pageLoad(idx, size, st);
	}

}

template<> void cacheControl<GPU_CUDA>::uncachePtr(void* ptr, size_t size, cudaStream_t* st, CACHE_TYPE CT) {
//	unsigned long addr_end = ((unsigned long) ptr + size) & address_filter;
//	for (unsigned long addr_curr = (unsigned long) ptr & address_filter;
//			addr_curr < addr_end; addr_curr += BLOCK_SIZE) {
//		unsigned long idx = (addr_curr & index_filter) >> BLOCK_BITS;
//		unsigned long value = __sync_val_compare_and_swap(
//				(volatile unsigned long*) &pages_local[idx], 0, addr_curr);
//
//		if (value == addr_curr) {
//			if (__sync_fetch_and_sub(
//					(volatile unsigned long*) &pages_inflight[idx], 1) == 1) {
//				__sync_val_compare_and_swap(
//						(volatile unsigned long*) &pages_local[idx], addr_curr,
//						0);
//					pageLoad(idx, 1, st);
//			}
//		}
//	}
}
