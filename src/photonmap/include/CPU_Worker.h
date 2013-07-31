/*
 * CPU_Worker.h
 *
 *  Created on: Oct 31, 2012
 *      Author: rr
 */

#ifndef CPU_WORKER_H_
#define CPU_WORKER_H_

#include "core.h"
#include "pointerfreescene.h"
#include "hitpoints.h"
#include "List.h"
#include "cuda_utils.h"
#include "renderEngine.h"
#include "lookupAcc.h"
#include <xmmintrin.h>
#include "Profiler.h"
#include "luxrays/accelerators/qbvhaccel.h"
#include "config.h"
#include "Worker.h"

class CPU_Worker;

class CPU_Worker: public Worker {
public:

	/**
	 * in this worker used for raybuffer size
	 */
	uint WORK_SIZE;

	lookupAcc* lookupA;

	RayBuffer *rayBuffer;

	Seed* seedBuffer;

	HitPoint* HPsPositionInfo;
	//HitPointRadianceFlux* HPsIterationRadianceFlux;

	CPU_Worker(uint device, Engine* engine_) {

		size_t buffer_size = 1024 * 256;

		uint c = max(cfg->hitPointTotal, cfg->photonsFirstIteration); //One seed per sample/hitpoint path and one seed per photon path

		seedBuffer = new Seed[c];

		for (uint i = 0; i < c; i++)
			seedBuffer[i] = mwc(i + device);

		WORK_SIZE = buffer_size;

		deviceID = device;

		rayBuffer = new RayBuffer(buffer_size);

#ifdef USE_HASHGRID
		lookupA = new HashGridLookup(cfg->hitPointTotal, cfg->rebuildHash);
#endif

#ifdef USE_KDTREE
		lookupA = new KdTree();
#endif
		setScene(engine_->ss);

		engine = engine_;

		InitHitPoints();

	}

	~CPU_Worker();

	void Start(bool buildHitPoints) {

		thread = new boost::thread(
				boost::bind(CPU_Worker::Entry, this, buildHitPoints));

	}

	inline uint GetWorkSize() {
		return WORK_SIZE;
	}

	BBox* GetHostBBox() {
		return lookupA->getBBox();
	}

	void resetRayBuffer(bool realloc = false) {

		rayBuffer->Reset();

		memset(rayBuffer->GetHitBuffer(), 0,
				sizeof(RayHit) * rayBuffer->GetSize());

		memset(rayBuffer->GetRayBuffer(), 0,
				sizeof(Ray) * rayBuffer->GetSize());

	}

//	HitPoint *GetHitPointInfo(const unsigned int index) {
//		return &(HPsPositionInfo)[index];
//
//	}

	HitPoint *GetHitPoint(const unsigned int index) {

		return &(HPsPositionInfo)[index];
	}

	void AdvanceEyePaths(RayBuffer *rayBuffer, EyePath *todoEyePaths,
			uint *eyePathIndexes);

	u_int64_t BuildPhotonMap(u_int64_t photontarget);

	void updateDeviceHitPointsInfo(bool toHost);
	void updateDeviceHitPointsFlux();
	void ResetDeviceHitPointsFlux();
	void ResetDeviceHitPointsInfo();
	void ProcessEyePaths();

	void InitHitPoints();

	void CommitIterationHitPoints(u_int64_t photonPerIteration);
	void MirrorHitPoints();

	float GetNonPAMaxRadius2();
	void SetNonPAInitialRadius2(float photonRadius2);

	void Intersect(RayBuffer *rayBuffer);
	void IntersectRay(const Ray *ray, RayHit *rayHit);
	void AccumulateFluxPPMPA(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxPPM(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxSPPM(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxSPPMPA(uint iteration, u_int64_t photonTraced);

	void GetSampleBuffer();

	void UpdateBBox() {

		// Calculate hit points bounding box
		//std::cerr << "Building hit points bounding box: ";

		BBox hitPointsbbox = BBox();

		for (unsigned int i = 0; i < cfg->hitPointTotal; ++i) {
			HitPoint *hp = GetHitPoint(i);

			if (hp->type == SURFACE)
				hitPointsbbox = Union(hitPointsbbox, hp->position);
		}

		SetBBox(hitPointsbbox);



	}

	void CopyLookupAccToDevice() {

	}

	void getDeviceHitpoints() {

	}

	void Render(bool buildHitPoints) {

//#if defined USE_PPMPA || defined USE_PPM
//		__BENCH.REGISTER("1. Build Hit Points");
//
//		if (buildHitPoints) {
//			BuildHitPoints(1);
//
//		}
//		__BENCH.STOP("1. Build Hit Points");
//
//#if defined USE_PPMPA
//		//engine->waitForHitPoints->wait();
//#endif
//#endif
		__p.reg("Process Iterations");

		engine->ProcessIterations(this, buildHitPoints);

		__p.stp("Process Iterations");

	}

	size_t getRaybufferSize() {
		return rayBuffer->GetSize();
	}

	void SetBBox(BBox hitPointsbbox) {
		lookupA->setBBox(hitPointsbbox);
	}

	size_t RaybufferAddRay(const Ray &ray) {
		return rayBuffer->AddRay(ray);
	}

	uint getRayBufferRayCount() {
		return rayBuffer->GetRayCount();
	}

	void AdvanceEyePaths(EyePath* todoEyePaths, uint* eyePathIndexes) {
		AdvanceEyePaths(rayBuffer, todoEyePaths, eyePathIndexes);
	}

	void IntersectRayBuffer() {
		Intersect(rayBuffer);
	}

	void UpdateQueryRangeLookupAcc(uint it) {

		double start = WallClockTime();

		float maxRad2 = GetCurrentMaxRadius2();

		lookupA->UpdateQueryRange(maxRad2, it, GetHitPoint(0));

		double elapsed = WallClockTime() - start;

		fprintf(stderr, "Device %d: It %d Lookup update time: %.3f\n",
				getDeviceID(), it, elapsed);

	}

	void BuildLookupAcc() {

		double start = WallClockTime();
		float maxRad2 = GetCurrentMaxRadius2();

		lookupA->Build(maxRad2, GetHitPoint(0));

		double elapsed = WallClockTime() - start;

		fprintf(stderr, "Device %d: Build Lookup time: %.3f\n", getDeviceID(),
				elapsed);
	}

	RayBuffer* GetRayBuffer() {
		return rayBuffer;
	}

	static void Entry(CPU_Worker *worker, bool buildHitPoints) {
		worker->Render(buildHitPoints);
	}

};

#endif /* CUDA_WORKER_H_ */
