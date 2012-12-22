/*
 * DemandScheduler.h
 *
 *  Created on: Aug 7, 2012
 *      Author: jbarbosa
 */

#ifndef DEMANDSCHEDULER_H_
#define DEMANDSCHEDULER_H_

#define umax(A,B) ((A > B) ? A : B)
#define umin(A,B) ((A < B) ? A : B)
const double pre = 1.0 / (double)(TOTAL_DEVICES);


class DemandScheduler {

protected:
	std::deque<work*> _queue;
	pthread_mutex_t _lock;

	Workqueue< work , DEVICE_QUEUE_SIZE, CPU_X86>* INBOX[TOTAL_DEVICES];

	unsigned long _assigned[TOTAL_DEVICES+1][WORK_TOTAL+1];

public:
	DemandScheduler();
	virtual ~DemandScheduler();

	void enqueueMultiple(List<work*>* l);
	void enqueue(work* l);
	work* dequeue();
	std::vector<work*>* request(unsigned devID, double elapse);

	__inline__ void setQueueDevice(unsigned int devID, Workqueue< work , DEVICE_QUEUE_SIZE, CPU_X86>* inbox) {
		INBOX[devID] = inbox;
	}
	__inline__ unsigned assignChunk(unsigned int devID) {
		if(_assigned[devID][WORK_TOTAL] == 0) return 1;
		return umax(SCHEDULER_MINIMUM, SCHEDULER_CHUNK * ((double)_assigned[devID][WORK_TOTAL] / (double)_assigned[TOTAL_DEVICES][WORK_TOTAL]));
	}

	__inline__ bool checkEligiblity(unsigned int devID, unsigned int wt) {
		double r = (double) rand() / (double)RAND_MAX;
		if((double)_assigned[TOTAL_DEVICES][wt] <= SCHEDULER_CHUNK || _assigned[devID][wt] == 0) return true;
		double probB =  (double)_assigned[devID][wt] / (double)_assigned[TOTAL_DEVICES][wt];
		return (probB >= r);
	}

	__inline__ bool isEmpty() {
		return _queue.empty();
	}
};

#endif /* DEMANDSCHEDULER_H_ */
