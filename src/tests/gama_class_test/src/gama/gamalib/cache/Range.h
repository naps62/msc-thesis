/*
 * Range.h
 *
 *  Created on: Oct 25, 2012
 *      Author: jbarbosa
 */

#ifndef RANGE_H_
#define RANGE_H_

#include <assert.h>

#ifndef MAX
#define MAX(A,B) (A > B) ? A : B
#endif

#ifndef MIN
#define MIN(A,B) (A < B) ? A : B
#endif


class Range {

public:
	unsigned long addr_start;
	unsigned long addr_end;

	Range() : addr_start(INVALID_REF), addr_end(INVALID_REF) {
	}

	Range(unsigned long _addr_start,unsigned long _addr_end) : addr_start(_addr_start), addr_end(_addr_end) {
	    //assert(addr_start<addr_end);
    }

	virtual ~Range() {
	}


	bool overlap(Range r) {
		return (addr_end >= r.addr_start || r.addr_end >= addr_start);
	}


	bool merge(Range r) {
		if(!overlap(r)) return false;
		addr_start = MIN(addr_start,r.addr_start);
		addr_end = MAX(addr_end,r.addr_end);
		return true;
	}

	bool isInvalid() {
		return (addr_start == INVALID_REF || addr_end == INVALID_REF);
	}

    std::vector<Range>* difference(Range r) {
    	std::vector<Range>* diff = new std::vector<Range>();
    	diff->push_back(r);
    	return diff;
    }

};



#endif /* RANGE_H_ */
