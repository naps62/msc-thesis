/*
 * Profiler.h
 *
 *  Created on: Nov 9, 2012
 *      Author: rr
 */

#ifndef PROFILER_H_
#define PROFILER_H_

#include <stdio.h>
#include <iostream>

class Profiler {
public:

	unsigned long long int photonsTraced;
	unsigned long long raysTraced;
	unsigned int iterationCount;

	double photonTracingTime;
	double rayTracingTime;
	double iteratingTime;

	Profiler() {
		photonsTraced = 0;
		raysTraced = 0;
		iterationCount = 0;

		photonTracingTime = 0.0f;
		rayTracingTime = 0.0f;
		iteratingTime = 0.0f;
	}
	virtual ~Profiler(){

	}

	void addPhotonsTraced(unsigned long long int p) {
		photonsTraced += p;
	}

	void addRaysTraced(unsigned long long p) {
		raysTraced += p;
	}

	void addIteration(unsigned int p) {
		iterationCount += p;
	}

	void addPhotonTracingTime(double p) {
		photonTracingTime += p;
	}

	void addRayTracingTime(double p) {
		rayTracingTime += p;
	}

	void additeratingTime(double p) {
		iteratingTime += p;
	}

	void printStats(uint id) {

		stringstream s;



		s  << "Device " << id << " stats:\n";

		float MPhotonsSec = photonsTraced / (photonTracingTime * 1000000.f);
		float itsec = iterationCount / iteratingTime;
		float MRaysSec = raysTraced / (rayTracingTime * 1000000.f);

		s  << "MPhotons/sec|" << MPhotonsSec <<  "\n";


		s  << "MRays/sec|"  << MRaysSec << "\n";
		s  << "iteration/sec|" << itsec << "\n";

		s  << "#####################\n";

		cout << s.str();


	}

};

#endif /* PROFILER_H_ */
