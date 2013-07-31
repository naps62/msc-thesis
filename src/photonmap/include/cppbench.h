/*
 * cbench.h
 *
 *  Created on: Jul 17, 2012
 *      Author: rr
 */

#ifndef CPPBENCH_H_
#define CPPBENCH_H_

#include <string>
#include <iostream>
#include <map>
#include <sys/time.h>

#define MAX_LEVELS 10
using namespace std;

inline double CPPBENCHWallClockTime() {
#if defined(__linux__) || defined(__APPLE__) || defined(__CYGWIN__)
	struct timeval t;
	gettimeofday(&t, NULL);

	return t.tv_sec + t.tv_usec / 1000000.0;
#elif defined (WIN32)
	return GetTickCount() / 1000.0;
#else
#error "Unsupported Platform !!!"
#endif
}

class CPPBENCH;

extern CPPBENCH __p;

class CPPBENCH {
public:

	map<string, double>* timers;
	map<string, uint>* insertion_order;

	uint lvl;

	CPPBENCH() {

		timers = new map<string, double>();
		insertion_order = new map<string, uint>();
		lvl = 0;

	}

	~CPPBENCH() {
	}

	void inline reg(string s) { //REGISTER

		insertion_order->insert(pair<string, uint>(s,lvl++));
		timers->insert(pair<string, double>(s, CPPBENCHWallClockTime()));
	}

	void inline stp(string s) { //STOP
		(*timers)[s] = CPPBENCHWallClockTime() - (*timers)[s];
	}

	void PRINT_SECONDS(string s) {

		printf("CBENCH |  %s: %.2f2s\n", s.c_str(), (*timers)[s]);

	}

	double GET_SECONDS(string s) {

		return ((*timers)[s]);

	}

	/*
	 * Prev + (end - star) = (Prev - start) + end
	 * And a start is always after nothing or a stop
	 */
	void inline lsstt(string s) { //LOOP_STAGE_START

		map<string, double>::iterator it;

		it = timers->find(s);

		if (it != timers->end()) {
			(*timers)[string(s)] -= CPPBENCHWallClockTime();
		} else {
			insertion_order->insert(pair<string, uint>(s,lvl++));
			timers->insert(pair<string, double>(s, -CPPBENCHWallClockTime()));
		}

	}

	void inline lsstp(string s) { //LOOP_STAGE_STOP

		(*timers)[s] += CPPBENCHWallClockTime();

	}

	void PRINTALL_SECONDS() {

		string* ordered = new string[timers->size()];
		map<string, double>::iterator it;

		for (it = timers->begin(); it != timers->end(); it++)
			ordered[insertion_order->at(it->first)] = it->first;

		printf(
				"\n\n=============================================================================\n");

		for (uint i = 0; i < timers->size(); i++)
			printf("%-67.67s | %-2.3f |\n", ordered[i].c_str(),
					timers->at(ordered[i]));
		printf(
				"=============================================================================\n");
	}

	// in progress...
	char* C_STR(string s) {
		char* buffer = new char[32];
		snprintf(buffer, 32, "%g", (*timers)[s] / 1000);
		return buffer;
	}

};

#endif

