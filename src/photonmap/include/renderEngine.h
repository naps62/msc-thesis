/*
 * smallppmgpu.h
 *
 *  Created on: Jul 25, 2012
 *      Author: rr
 */

#ifndef SMALLPPMGPU_H_
#define SMALLPPMGPU_H_

#include "core.h"
#include "pointerfreescene.h"
#include "hitpoints.h"
#include "List.h"
#include "config.h"
#include "RenderConfig.h"
#include "Worker.h"
#include "omp.h"

class PhotonPath;
//class Worker;

void InitGlut(int argc, char *argv[], const unsigned int width,
		const unsigned int height);

void RunGlut(const unsigned int width, const unsigned int height);

class Engine {

public:
	//boost::mutex hitPointsLock;

	boost::thread* draw_thread;

	PointerFreeScene *ss;

//	HitPointPositionInfo* scatch_HPsPositionInfo;
//	HitPointRadianceFlux* scatch_HPsIterationRadianceFlux;

private:
//unsigned long long photonTraced;
	uint interationCount;
	Film *film;
	unsigned long long photonTracedTotal;

public:

	Engine() {

//		scatch_HPsIterationRadianceFlux =
//				new HitPointRadianceFlux[cfg->hitPointTotal];
//		memset(scatch_HPsIterationRadianceFlux, 0,
//		sizeof(HitPointRadianceFlux) * cfg->hitPointTotal);
//
//		scatch_HPsPositionInfo = new HitPointPositionInfo[cfg->hitPointTotal];
//		memset(scatch_HPsPositionInfo, 0,
//				sizeof(HitPointPositionInfo) * cfg->hitPointTotal);

		interationCount = 1;

		photonTracedTotal = 0;

		film = new Film(cfg->width, cfg->height);
		film->Reset();

	}

	virtual ~Engine();

	unsigned long long incPhotonTracedTotal(unsigned long long i) {
		return __sync_fetch_and_add(&photonTracedTotal, i);
	}

	unsigned long long getPhotonTracedTotal() {
		return photonTracedTotal;
	}

	uint GetIterationNumber() {
		return interationCount;
	}

	uint IncIteration() {
		return interationCount++;
	}

	bool GetHitPointInformation(PointerFreeScene *ss, Ray *ray,
			const RayHit *rayHit, Point & hitPoint, Spectrum & surfaceColor,
			Normal & N, Normal & shadeN);

	void InitPhotonPath(PointerFreeScene *ss, PhotonPath *photonPath, Ray *ray,
			Seed& seed);

	void SplatSampleBuffer(SampleFrameBuffer* sampleFrameBuffer,
			const bool preview, SampleBuffer *sampleBuffer) {

		film->SplatSampleBuffer(sampleFrameBuffer, preview, sampleBuffer);
	}

	void ResetFilm() {
		film->Reset();
	}

	void UpdateScreenBuffer() {
		film->UpdateScreenBuffer();
	}

	void LockImageBuffer() {
		film->imageBufferMutex.lock();
	}

	void UnlockImageBuffer() {
		film->imageBufferMutex.unlock();
	}

	bool filmCreated() {
		return film;
	}

	const float * GetScreenBuffer() {
		return film->GetScreenBuffer();
	}

	void SaveImpl(const std::string &fileName) {
		film->SaveImpl(fileName);

	}

	void UpdateSampleFrameBuffer(unsigned long long iterationPhotonCount,
			Worker* w) {

		w->GetSampleBuffer();
		w->sampleFrameBuffer->Clear();
		__p.lsstt(
				"Process Iterations > Iterations > Update Samples > Splat to pixel");

		SplatSampleBuffer(w->sampleFrameBuffer, true, w->sampleBuffer);

		__p.lsstp(
				"Process Iterations > Iterations > Update Samples > Splat to pixel");

	}

	virtual void ProcessIterations(Worker* worker, bool buildHitPoints)=0;
	virtual void InitRadius(uint iteration, Worker* w)=0;

	virtual HitPoint *GetHitPoints() {
		assert(false);
		return NULL;
	}

};

/**
 * PPM
 * Single device.
 * Dependant iterations, single build hitpoints, reduce radius and reflected flux.
 * Radius per iteration, dependant and per hit point.
 * Keep local statistics.
 */
class PPMEngine: public Engine {
public:

	PPMEngine() {
		//HPsPositionInfo_central = new HitPointPositionInfo[cfg->hitPointTotal];

	}

	virtual ~PPMEngine();

//	HitPointPositionInfo *GetHitPoints() {
//		return HPsPositionInfo_central;
//	}

	void ProcessIterations(Worker* worker, bool buildHitPoints);
	void InitRadius(uint iteration, Worker* w);

	//HitPointPositionInfo* HPsPositionInfo_central;

};

/**
 * SPPM
 * Single device.
 * Dependant iterations, in each iterations build hitpoints, reduce radius and reflected flux.
 * Radius per iteration, dependant and per hit point.
 * Keep local statistics.
 */

class SPPMEngine: public Engine {
public:

	SPPMEngine() {
	}

	virtual ~SPPMEngine() {
	}

	void ProcessIterations(Worker* worker, bool buildHitPoints);
	void InitRadius(uint iteration, Worker* w);
};

/**
 * PPM:PA
 * Single hit points, each device mirrors hpts and builds hash grid.
 * Iterations independent, radius not reduced.
 * Oversampling.
 * Multi-resolution grid targeted.
 */
class PPMPAEngine: public Engine {
public:
	PPMPAEngine() {

		HPsPositionInfo_central = new HitPoint[cfg->hitPointTotal];

	}

	virtual ~PPMPAEngine() {
	}

	void ProcessIterations(Worker* worker, bool buildHitPoints);
	void InitRadius(uint iteration, Worker* w);

	HitPoint *GetHitPoints() {
		return HPsPositionInfo_central;
	}

	boost::barrier* waitForHitPoints;
	HitPoint* HPsPositionInfo_central;

};

/**
 * SPPM:PA
 * Each device builds hitpoints and hash.
 * Iterations independent, radius not reduced -> precalculated.
 * Radius per iteration, not per hitpoint.
 * 1 inital SPP.
 * Paper PPM:PA approach reversed.
 */
class SPPMPAEngine: public Engine {
public:

	SPPMPAEngine() {
	}

	virtual ~SPPMPAEngine() {
	}

	void ProcessIterations(Worker* worker, bool buildHitPoints);
	void InitRadius(uint iteration, Worker* w);

};

#endif /* SMALLPPMGPU_H_ */
