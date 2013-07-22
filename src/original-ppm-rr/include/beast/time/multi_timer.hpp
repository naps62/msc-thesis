/*
 * multi_timer.hpp
 *
 *  Created on: Jan 11, 2013
 *      Author: naps62
 */

#ifndef _BEAST_MULTI_TIMER_HPP_
#define _BEAST_MULTI_TIMER_HPP_

#include <beast/common.hpp>
#include <beast/time/timer.hpp>

#include <vector>
using std::vector;

#include <string>
using std::string;

namespace beast { namespace time {

	/*
	 *
	 */
	class multi_timer : public timer {

	private:
		vector<double> _history;

	public:

		/*
		 * constructors
		 */
		multi_timer()
			: timer()
		{ }

		multi_timer(const string& name)
			: timer(name)
		{ }

		multi_timer(const char* name)
			: timer(name)
		{ }

		/*
		 * control functions
		 */
		__beast_force_inline__
		void stop() {
			if (running) {
				elapsed();
				running = false;
				_history.push_back(_elapsed.ns());
			}
		}

		/*
		 * history functions
		 */
		__beast_force_inline__
		int count() const {
			return _history.size();
		}

		double average() {
			return accumulate(_history.begin(), _history.end(), 0.0) / _history.size();
		}

		double median() {
			int middle = _history.size() / 2;
			nth_element(_history.begin(), _history.begin()+middle, _history.end());
			return _history[middle];
		}
	};

} }


#endif // _BEAST_MULTI_TIMER_HPP_
