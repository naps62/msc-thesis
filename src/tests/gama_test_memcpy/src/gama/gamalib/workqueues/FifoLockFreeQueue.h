#ifndef LOCKFREEQUEUE_H_
#define LOCKFREEQUEUE_H_

#include <pthread.h>
#include <deque>

template <typename T> class FIFOLockFreeQueue {
private:
	pthread_mutex_t _lock;

	volatile unsigned int _SpinLock;

public:

	std::deque<T> dq;
	FIFOLockFreeQueue() : _SpinLock(0) {
	}

	~FIFOLockFreeQueue(){
	}

	// Add new data to the queue
	void enqueue(const T &rNew){
		if(rNew == NULL) return;
		dq.push_back(rNew);
	}

	// Remove data from the queue if it is available
	bool dequeue(T& rValue){
		rValue = NULL;
		if(dq.empty()) return false;
		rValue = dq.front();
		dq.pop_front();
		return true;
	}

	bool isEmpty() {
		return dq.empty();//(pDivider == pLast);
	}

	T elem(int i) {
		return dq[i];
	}

	int size() {
		return dq.size();//(pDivider == pLast);
	}

	__inline bool trylock() {
		bool res = __sync_bool_compare_and_swap(&_SpinLock,0, 1);
		asm volatile("lfence" ::: "memory");
		//__sync_synchronize();
		return res;
	}

	__inline void release() {
		_SpinLock = 0 ;
		asm volatile("lfence" ::: "memory");
		//__sync_synchronize();
	}

};

#endif /* LOCKFREEQUEUE_H_ */
