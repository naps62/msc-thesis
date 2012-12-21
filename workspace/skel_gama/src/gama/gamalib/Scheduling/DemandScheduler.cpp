/*
 * DemandScheduler.cpp
 *
 *  Created on: Aug 7, 2012
 *      Author: jbarbosa
 */
#include <config/common.h>

#include <deque>
#include <map>
#include <vector>
#include <gamalib/gamalib.h>
#include <pthread.h>

#include "DemandScheduler.h"

DemandScheduler::DemandScheduler() {
	pthread_mutex_init(&_lock,NULL);

    printf("(SYS) Demand Driven\n");

	for(unsigned dev=0; dev < TOTAL_DEVICES+1; dev++) {
		for(unsigned wt=0; wt < WORK_TOTAL+1; wt++) {
			_assigned[dev][wt] = (dev != TOTAL_DEVICES) ? 0 : 1;
			if(wt == WORK_TOTAL) _assigned[dev][wt] = 0;
		}
	}
}

DemandScheduler::~DemandScheduler() {
	for(int dev=0; dev < TOTAL_DEVICES+1; dev++ ) {

		if(dev < TOTAL_DEVICES) {
			printf("Device %s \t",(DEVICE_TYPES[dev] == CPU_X86) ? "CPU" : "GPU");
		} else {
			printf("Total  %s \t","HYB");
		}

		for(unsigned wt=0; wt < WORK_TOTAL+1; wt++)
			printf("%8lu\t", _assigned[dev][wt] );
		printf("\n");
	}
}

void DemandScheduler::enqueueMultiple(List<work*>* l) {
	pthread_mutex_lock(&_lock);
	for(int ii=0; ii < l->getSize(); ii++) _queue.push_back((*l)[ii]);
	X86MemFence();
	pthread_mutex_unlock(&_lock);
}

void DemandScheduler::enqueue(work* w) {
	pthread_mutex_lock(&_lock);
	_queue.push_back(w);
	pthread_mutex_unlock(&_lock);
}

work* DemandScheduler::dequeue() {
	work* ret = NULL;
	pthread_mutex_lock(&_lock);
	if(!_queue.empty()) {
		ret = _queue.front();

		_queue.pop_front();
	}
	pthread_mutex_unlock(&_lock);
	return ret;
}

std::vector<work*>* DemandScheduler::request(unsigned devID, double elapse) {
	std::vector<work*>* ret = new std::vector<work*>();;
	//if(_queue.empty()) return ret;
	if(pthread_mutex_trylock(&_lock) == 0) {
		unsigned CHUNK = SCHEDULER_CHUNK; //assignChunk(devID);
		if(!_queue.empty()) {
			for(; CHUNK > 0 && !_queue.empty(); CHUNK--) {
				work* w = _queue.front();
				_queue.pop_front();
				if(checkEligiblity(devID,w->getWorkTypeID())) {
					ret->push_back(w);
					_assigned[devID][w->getWorkTypeID()]++;
					_assigned[devID][WORK_TOTAL]++;
					_assigned[TOTAL_DEVICES][w->getWorkTypeID()]++;
					_assigned[TOTAL_DEVICES][WORK_TOTAL]++;
				} else {
					if(w != NULL) _queue.push_back(w);
					break;
				}
			}
		} else {
			unsigned int victim = rand() % TOTAL_DEVICES;
			if(CHUNK > 0 && victim != devID && INBOX[victim]->getSize() > SCHEDULER_MINIMUM) {
				for(; 0 < CHUNK; CHUNK--) {
					work* w = NULL;
					INBOX[victim]->dequeueBack(w);
					if(w != NULL && checkEligiblity(devID,w->getWorkTypeID())) {
						ret->push_back(w);
						_assigned[devID][w->getWorkTypeID()]++;
						_assigned[devID][WORK_TOTAL]++;
						_assigned[TOTAL_DEVICES][w->getWorkTypeID()]++;
						_assigned[TOTAL_DEVICES][WORK_TOTAL]++;
					} else {
						if(w !=NULL) INBOX[victim]->enqueue(w);
						break;
					}
				}

			}
		}
		pthread_mutex_unlock(&_lock);
	}
	return ret;
}

