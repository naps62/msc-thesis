/*
 * time_val.hpp
 *
 *  Created on: Jan 6, 2013
 *      Author: naps62
 */

#ifndef _BEAST_TIME_VAL_HPP_
#define _BEAST_TIME_VAL_HPP_

#include <beast/common.hpp>

#include <ctime>

namespace beast { namespace time {

	/*
	 * stores a time value
	 */
	struct time_val : public timespec {

		/*
		 * constructors
		 */

		// default constructor. sets value to 0
		time_val();

		// constructor with initial time value (by pointer)
		time_val(const time_val* const t);

		// constructor with initial time value (by reference)
		time_val(const time_val& t);

		// resets the value to 0
		void zero();

		/*
		 * getters
		 */
		__beast_force_inline__ double ns() const { return tv_sec * 1e9 + tv_nsec; }
		__beast_force_inline__ double us() const { return ns() * 1e-3; }
		__beast_force_inline__ double ms() const { return ns() * 1e-6; }
		__beast_force_inline__ double s () const { return ns() * 1e-9; }

		/*
		 * operators
		 */
		const time_val& operator-= (const time_val& t);
			  time_val  operator-  (const time_val& t) const;
		const time_val& operator+= (const time_val& t);
		const time_val  operator+  (const time_val& t);
	};
} }

#endif // _BEAST_TIME_VAL_HPP_
