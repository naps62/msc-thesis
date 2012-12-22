/*
 * cacheControl.h
 *
 *  Created on: Sep 26, 2012
 *      Author: jbarbosa
 */

#ifndef CACHECONTROL_H_
#define CACHECONTROL_H_

#include <config/common.h>
#include "CacheList.h"

template<DEVICE_TYPE T>
class cacheControl {
public:

	void* cache;
	unsigned long* pages_cache;
	unsigned long* pages_local;
	unsigned long* pages_inflight;
	unsigned int* _lock;

	double total_time;
	volatile unsigned int _lock_all;
	volatile unsigned long _copies_in_flight;

	CacheList cl;
    CacheList cached;

	cacheControl() : cache(0), pages_cache(0), pages_local(0), pages_inflight(0), _lock(0), _lock_all(0), _copies_in_flight(0), total_time(0.f){
	}

	virtual ~cacheControl();

	void cachePtr(void* ptr, size_t size,cudaStream_t* st, CACHE_TYPE CT=CACHE_READ_ONLY) {}
	void uncachePtr(void* ptr, size_t size,cudaStream_t* st, CACHE_TYPE CT=CACHE_READ_ONLY) {}
	void cacheReset();
	void cacheMove(unsigned long, unsigned long,cudaStream_t* st);
	void pageLoad(unsigned long, unsigned long, cudaStream_t* st);
	void emptyLocked(std::vector<unsigned long> &locked) {};
};


template<> cacheControl<GPU_CUDA>::cacheControl();
template<> cacheControl<GPU_CUDA>::~cacheControl();

template<> void cacheControl<GPU_CUDA>::cachePtr(void* ptr, size_t size,cudaStream_t* st, CACHE_TYPE CT);
template<> void cacheControl<GPU_CUDA>::uncachePtr(void* ptr, size_t size, cudaStream_t* st, CACHE_TYPE CT);

template<> void cacheControl<GPU_CUDA>::cacheReset();
template<> void cacheControl<GPU_CUDA>::cacheMove(unsigned long, unsigned long,cudaStream_t* st);
template<> void cacheControl<GPU_CUDA>::pageLoad(unsigned long, unsigned long, cudaStream_t* st);

template<> void cacheControl<GPU_CUDA>::emptyLocked(std::vector<unsigned long> &locked);

#endif /* CACHECONTROL_H_ */
