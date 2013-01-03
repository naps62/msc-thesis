/*
 * prof_tottime.h
 *
 *  Created on: Jan 3, 2013
 *      Author: naps62
 */

#ifndef PROF_TOTTIME_H_
#define PROF_TOTTIME_H_

#include <tk/stopwatch-inl.hpp>

#define PROFILE_LIMITED 1000
#define PROFILE_WARMUP   100

#define PROFILE_COUNTER_CLASS tk::Stopwatch
#define PROFILE_COUNTER_NAME  s
#define PROFILE_COUNTER       profile::PROFILE_COUNTER_NAME



namespace profile
{
	PROFILE_COUNTER_CLASS * PROFILE_COUNTER_NAME;

	long long int mntotus;

	inline
	void init()
	{
		PROFILE_COUNTER_NAME = new PROFILE_COUNTER_CLASS();

		mntotus = 0;

		PROFILE_COUNTER_NAME->start();
	}

	inline
	void output(std::ostream& out)
	{
		PROFILE_COUNTER_NAME->stop();

		mntotus += PROFILE_COUNTER_NAME->last().microseconds();

		out
			<<	mntotus	<<	';'
								<<	std::endl
			;
	}

	void cleanup()
	{
		delete PROFILE_COUNTER_NAME;
	}
}



#define PROFILE_INIT() profile::init()

#define PROFILE_RETRIEVE_CF() ;

#define PROFILE_RETRIEVE_UP() ;

#define PROFILE_OUTPUT() profile::output(std::cout)

#define PROFILE_CLEANUP()

#define PROFILE


#define PROFILE_START() {;}

#define PROFILE_STOP() {;}


#endif /* PROF_TOTTIME_H_ */
