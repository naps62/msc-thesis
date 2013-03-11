/*
 * CacheList.h
 *
 *  Created on: Oct 25, 2012
 *      Author: jbarbosa
 */

#ifndef CACHELIST_H_
#define CACHELIST_H_

#include "Range.h"
#include <list>

class CacheList {

	std::list<Range> list;
	unsigned int _lock;
public:
	CacheList() {
		list = std::list<Range>();
	}
	virtual ~CacheList() {

	}

	void insert(Range r) {

		while(__sync_val_compare_and_swap(&_lock,0,1) !=0);

		if (list.empty()) {
			list.push_front(r);
			__sync_val_compare_and_swap(&_lock,1,0);
			return;
		}
		std::list<Range>::iterator it;
		bool found = false;
		for (it = list.begin(); !(found=it->merge(r)) && it != list.end(); it++);
		if(!found) list.push_front(r);
		__sync_val_compare_and_swap(&_lock,1,0);
		return;
	}

	std::list<Range>& getList() {
		return list;
	}

	bool empty() {
		return list.empty();
	}

	void clear() {
		list.clear();
	}

	Range next() {
		while(__sync_val_compare_and_swap(&_lock,0,1) !=0);
		if(list.empty()) {
			__sync_val_compare_and_swap(&_lock,1,0);
			return Range(INVALID_REF,INVALID_REF);
		}
		Range r = list.back();
		list.pop_back();
		__sync_val_compare_and_swap(&_lock,1,0);
		return r;


	}

    std::vector<Range>* difference(Range r) {
   
        std::vector<Range>* ret = new std::vector<Range>();
        ret->push_back(r);
        std::list<Range>::iterator lit;

        for(lit=list.begin(); lit != list.end(); lit++) {
            Range rl = *lit;
        }
        return ret;
    }


};

#endif /* CACHELIST_H_ */
