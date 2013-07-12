/*
 * Worker.h
 *
 *  Created on: Nov 10, 2012
 *      Author: rr
 */

#ifndef WORKER_H_
#define WORKER_H_

#include "core.h"
#include "renderEngine.h"
#include "config.h"
#include "Profiler.h"

class Worker {
public:

	std::thread* thread;
	uint deviceID;

	PointerFreeScene *ss;

	float currentPhotonRadius2;

	HitPointStaticInfo* hitPointsStaticInfo_iterationCopy;
	HitPoint* hitPoints_iterationCopy;

	Profiler* profiler;
	SampleBuffer *sampleBuffer;

	SampleFrameBuffer *sampleFrameBuffer;

	Seed* seedBuffer;

	Worker(Seed* s) {

		seedBuffer = s;

		sampleBuffer = new SampleBuffer(
						engine->width * engine->height * engine->superSampling * engine->superSampling);

		sampleFrameBuffer = new SampleFrameBuffer(engine->width, engine->height);
				sampleFrameBuffer->Clear();


		profiler = new Profiler();

		hitPointsStaticInfo_iterationCopy = new HitPointStaticInfo[engine->hitPointTotal];
		hitPoints_iterationCopy = new HitPoint[engine->hitPointTotal];


		memset(hitPoints_iterationCopy, 0, sizeof(HitPoint) * engine->hitPointTotal);
		memset(hitPointsStaticInfo_iterationCopy, 0, sizeof(HitPointStaticInfo) * engine->hitPointTotal);
	}
	virtual ~Worker();


	void BuildHitPoints(uint iteration);

	void UpdateBBox();

	void ProcessIterations(PPM *engine);
	void InitRadius( uint iteration);
	void UpdateSampleFrameBuffer(unsigned long long iterationPhotonCount);

	void AccumulateFluxPPM(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxSPPM(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxSPPMPA(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxPPMPA(uint iteration, u_int64_t photonTraced);

	HitPointStaticInfo *GetHitPointInfo(const unsigned int index);
	HitPoint *GetHitPoint(const unsigned int index);

	void setScene(PointerFreeScene *s);
	uint getDeviceID();


//	void PushHitPoints();
//	void PullHitPoints();

	/**
	 * Centralized in the lookup table class. always updated there.
	 */
	virtual BBox* GetHostBBox() =0;
	virtual void IntersectRayBuffer()=0;
	virtual size_t getRaybufferSize()=0;
	virtual uint getRayBufferRayCount()=0;
	virtual size_t RaybufferAddRay(const Ray &ray) =0;
	virtual void AdvanceEyePaths(EyePath* todoEyePaths,uint* eyePathIndexes)=0;
	virtual void resetRayBuffer()=0;
	virtual u_int64_t AdvancePhotonPath( u_int64_t photontarget)=0;
	virtual void updateDeviceHitPoints()=0;
	virtual void updateDeviceLookupAcc()=0;
	virtual void getDeviceHitpoints()=0;
	virtual void ReHash(float currentPhotonRadius2)=0;
	virtual RayBuffer* GetRayBuffer()=0;


	virtual void SetBBox(BBox hitPointsbbox)=0;
	virtual void LookupSetHitPoints(HitPointStaticInfo* iterationHitPoints,HitPoint* workerHitPoints)=0;

};

#endif /* WORKER_H_ */

