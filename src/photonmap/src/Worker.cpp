/*
 * Worker.cpp
 *
 *  Created on: Nov 10, 2012
 *      Author: rr
 */

#include "Worker.h"
#include "omp.h"

Worker::~Worker() {
	// TODO Auto-generated destructor stub
}

void Worker::UpdateProfiler(uint iterationCount, double start) {

	profiler->additeratingTime(WallClockTime() - start);
	profiler->addIteration(1);

	//if (profiler->iterationCount % 10 == 0)
	//	profiler->printStats(deviceID);

}

void Worker::BuildHitPoints(uint iteration) {

//	if (cfg->GetEngineType() != SPPM)
//		ResetDeviceHitPointsFlux();

//	if (cfg->GetEngineType() == SPPM || cfg->GetEngineType() == SPPMPA)
//		ResetDeviceHitPointsInfo();

	ProcessEyePaths();

}

//HitPointPositionInfo *Worker::GetHitPointInfo(const unsigned int index) {
//	return &(HPsPositionInfo)[index];
//
//}
//
//HitPointRadianceFlux *Worker::GetHitPoint(const unsigned int index) {
//
//	return &(HPsIterationRadianceFlux)[index];
//}

void Worker::setScene(PointerFreeScene *s) {
	ss = s;
}

uint Worker::getDeviceID() {
	return deviceID;
}

/**
 * GPU?
 */
float Worker::GetCurrentMaxRadius2() {

	float maxPhotonRadius2;
	if (cfg->GetEngineType() == PPMPA || cfg->GetEngineType() == SPPMPA)
		maxPhotonRadius2 = currentPhotonRadius2;
	else
		maxPhotonRadius2 = GetNonPAMaxRadius2();

	return maxPhotonRadius2;
}
