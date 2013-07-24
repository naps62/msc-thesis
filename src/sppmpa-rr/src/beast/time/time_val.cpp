/*
 * time_val.cpp
 *
 *  Created on: Mar 12, 2013
 *      Author: naps62
 */

#include <beast/time/time_val.hpp>

namespace beast { namespace time {

	/*
	 * constructors
	 */

	// default constructor. sets value to 0
	time_val :: time_val() {
		zero();
	}

	// constructor with initial time value (by pointer)
	time_val :: time_val(const time_val* const t) {
		this->tv_sec  = t->tv_sec;
		this->tv_nsec = t->tv_nsec;
	}

	// constructor with initial time value (by reference)
	time_val :: time_val(const time_val& t) {
		this->tv_sec  = t.tv_sec;
		this->tv_nsec = t.tv_nsec;
	}

	// resets the value to 0
	void time_val :: zero() {
		tv_sec = tv_nsec = 0;
	}


	/*
	 * operators
	 */
	const time_val& time_val :: operator-= (const time_val& t) {
		if (t.tv_sec > tv_sec || (t.tv_sec == tv_sec && t.tv_nsec > tv_nsec)) {
			zero();
		} else {
			tv_sec -= t.tv_sec;
			tv_nsec -= t.tv_nsec;
			if (tv_nsec < 0) {
				--tv_nsec;
				tv_nsec += 1e9;
			}
		}
		return *this;
	}

	time_val time_val :: operator- (const time_val& t) const {
		time_val result(*this);
		result -= t;
		return result;
	}

	const time_val& time_val :: operator+= (const time_val& t) {
		tv_sec  += t.tv_sec;
		tv_nsec += t.tv_nsec;
		if (tv_nsec > 1e9) {
			++tv_sec;
			tv_nsec -= 1e9;
		}
		return *this;
	}

	const time_val time_val :: operator+ (const time_val& t) {
		time_val result(*this);
		result += t;
		return result;
	}

} }
