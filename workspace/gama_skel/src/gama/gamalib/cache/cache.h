/*
 * cache.h
 *
 *  Created on: Sep 26, 2012
 *      Author: jbarbosa
 */

#ifndef CACHE_H_
#define CACHE_H_

#ifdef __CUDACC__

__DEVICE__ __forceinline void* translateAddrGPU(void* ptr) {
#ifdef GAMA_CACHE
	unsigned long addr = ((unsigned long)ptr & address_filter);
	unsigned long idx = ((unsigned long)ptr & index_filter) >> BLOCK_BITS;
	if (_pages[idx] != addr) {
        return ptr; 
    } else {
        return (void*)((unsigned long)_cache + ((unsigned long)ptr & offset_filter));
    }
#else
	return ptr;
#endif
}

#define translateAddr(PTR) translateAddrGPU(PTR)

#else
__DEVICE__ __forceinline void* translateAddrCPU(void* ptr) {
	return ptr;
}

#define translateAddr(PTR) translateAddrCPU(PTR)

#endif
#endif /* CACHE_H_ */
