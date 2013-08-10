/*
 * timer.cpp
 *
 *  Created on: Mar 12, 2013
 *      Author: naps62
 */

#include <beast/time/timer.hpp>

namespace beast { namespace time {

	/*
	 * constructors
	 */

	// default constructor
	timer :: timer()
	: _name(BEAST_TIMER_NAME), running(false)
	{ }

	// constructor with a name (via string)
	timer :: timer(const string& name)
	: _name(name), running(false)
	{ }

	// constructor with a name (via char*)
	timer :: timer(const char* name)
	: _name(name), running(false)
	{ }

} }
