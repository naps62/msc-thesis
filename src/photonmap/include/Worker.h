/*
 * Worker.h
 *
 *  Created on: Nov 10, 2012
 *      Author: rr
 */

#ifndef WORKER_H_
#define WORKER_H_

#include "core.h"
#include "config.h"
#include "cppbench.h"
#include "pointerfreescene.h"
#include "hitpoints.h"
#include "RenderConfig.h"
#include "Profiler.h"

class Worker {
public:
	boost::thread* thread;
	uint deviceID;

	PointerFreeScene *ss;

	float currentPhotonRadius2;

//	HitPointPositionInfo* HPsPositionInfo;
//	HitPointRadianceFlux* HPsIterationRadianceFlux;

	Profiler* profiler;
	SampleBuffer *sampleBuffer;

	SampleFrameBuffer *sampleFrameBuffer;


	Engine* engine;

	Worker() {

		sampleBuffer = new SampleBuffer(
				cfg->width * cfg->height * cfg->superSampling
						* cfg->superSampling);

		sampleFrameBuffer = new SampleFrameBuffer(cfg->width, cfg->height);
		sampleFrameBuffer->Clear();

		profiler = new Profiler();

	}

	void BuildHitPoints(uint iteration);
	void UpdateProfiler(uint iterationCount, double start);
	//HitPointPositionInfo *GetHitPointInfo(const unsigned int index);
	//HitPointRadianceFlux *GetHitPoint(const unsigned int index);
	void setScene(PointerFreeScene *s);
	uint getDeviceID();

	virtual ~Worker();
	virtual void UpdateBBox()=0;
	virtual void Start(bool buildHitPoints = false)=0;
	virtual void AccumulateFluxPPM(uint iteration, u_int64_t photonTraced) = 0;
	virtual void AccumulateFluxSPPM(uint iteration, u_int64_t photonTraced) = 0;
	virtual void AccumulateFluxSPPMPA(uint iteration,
			u_int64_t photonTraced) = 0;
	virtual void AccumulateFluxPPMPA(uint iteration,
			u_int64_t photonTraced) = 0;

	virtual float GetCurrentMaxRadius2();

	virtual void GetSampleBuffer()=0;

	/**
	 * Centralized in the lookup table class. always updated there.
	 */
	virtual BBox* GetHostBBox() =0;
	virtual void ProcessEyePaths()=0;
	virtual u_int64_t BuildPhotonMap(u_int64_t photontarget)=0;
	virtual void updateDeviceHitPointsInfo(bool toHost)=0;
	virtual void ResetDeviceHitPointsInfo()=0;
	virtual float GetNonPAMaxRadius2()=0;
	virtual void SetNonPAInitialRadius2(float photonRadius2)=0;
	virtual void UpdateQueryRangeLookupAcc(uint it)=0;
	virtual void BuildLookupAcc()=0;
	virtual void CopyLookupAccToDevice()=0;
	virtual void getDeviceHitpoints()=0;
//	virtual void InitHitPoints()=0;
	virtual void SetBBox(BBox hitPointsbbox)=0;

};

#endif /* WORKER_H_ */

