/*
 * timer.hpp
 *
 *  Created on: Jan 6, 2013
 *      Author: naps62
 */

#ifndef _BEAST_TIMER_HPP_
#define _BEAST_TIMER_HPP_

#include <beast/common.hpp>
#include <beast/time/defaults.hpp>
#include <beast/time/time_val.hpp>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <numeric>
using std::accumulate;

#include <algorithm>
using std::nth_element;

#include <iostream>
using namespace std;

namespace beast { namespace time {

	class timer {

	protected:

		const string _name;
		bool running;
		time_val _start;
		time_val _end;
		time_val _elapsed;


	public:

		/*
		 * constructors
		 */

		// default constructor
		timer();

		// constructor with a name (via string)
		timer(const string& name);

		// constructor with a name (via char*)
		timer(const char* name);

		/*
		 * control functions
		 */

		// starts the timer
		__beast_force_inline__
		void start() {
			running = true;
			gettime(_start);
		}

		// stops the timer
		__beast_force_inline__
		void stop() {
			if (running) {
				elapsed();
				running = false;
			}
		}

		// returns the amount of time elapsed since last start
		// (or until current time if timer is still running)
		__beast_force_inline__
		time_val elapsed() {
			if (running) {
				gettime(_end);
				_elapsed = _end - _start;
			}
			return _elapsed;
		}

		/*
		 * getters
		 */
		__beast_force_inline__ string name()  const { return _name; }
		__beast_force_inline__ double ns()    const { return _elapsed.ns(); }
		__beast_force_inline__ double us()    const { return _elapsed.us(); }
		__beast_force_inline__ double ms()    const { return _elapsed.ms(); }
		__beast_force_inline__ double s ()    const { return _elapsed.s(); }
		__beast_force_inline__ double value() const { return ns(); }

	protected:

		__beast_force_inline__
		void gettime(timespec& t) {
			clock_gettime(BEAST_TIMER_CLOCK_TYPE, &t);
		}

	};

} }

#endif /* _BEAST_TIMER_HPP_ */
