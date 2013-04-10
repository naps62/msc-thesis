/*
 * Worker.cpp
 *
 *  Created on: Nov 10, 2012
 *      Author: rr
 */

#include "Worker.h"
#include "omp.h"

//extern uint _num_threads;
//extern uint _num_iters;
extern const Config* config;

Worker::~Worker() {
	// TODO Auto-generated destructor stub
}

void Worker::BuildHitPoints(uint /*iteration*/) {

	const unsigned int width = engine->width;
	const unsigned int height = engine->height;
	const unsigned int superSampling = engine->superSampling;

	const unsigned int hitPointTotal = engine->hitPointTotal;

	//Seed* EyePathsSeeds = new Seed[hitPointTotal];

	EyePath* todoEyePaths = new EyePath[hitPointTotal];

	memset(hitPointsStaticInfo_iterationCopy, 0, sizeof(HitPointStaticInfo) * engine->hitPointTotal);

#ifndef USE_SPPM
	memset(hitPoints_iterationCopy, 0, sizeof(HitPoint) * engine->hitPointTotal);
#endif

	unsigned int hitPointsIndex = 0;

	// Generate eye rays
	//std::cerr << "Building eye paths rays with " << superSampling << "x"
	//		<< superSampling << " super-sampling:" << std::endl;
	//std::cerr << "  0/" << height << std::endl;

//	double lastPrintTime = WallClockTime();
	const float invSuperSampling = 1.f / superSampling;

	for (unsigned int y = 0; y < height; ++y) {

//		if (WallClockTime() - lastPrintTime > 2.0) {
//			std::cerr << "  " << y << "/" << height << std::endl;
//			lastPrintTime = WallClockTime();
//		}

		//for all hitpoints
		for (unsigned int x = 0; x < width; ++x) {
			for (unsigned int sy = 0; sy < superSampling; ++sy) {
				for (unsigned int sx = 0; sx < superSampling; ++sx) {

					EyePath *eyePath = &todoEyePaths[hitPointsIndex];

					//EyePathsSeeds[hitPointsIndex] = mwc(hitPointsIndex + deviceID);


					eyePath->scrX = x + (sx + getFloatRNG(seedBuffer[hitPointsIndex]))
							* invSuperSampling - 0.5f;

					eyePath->scrY = y + (sy + getFloatRNG(seedBuffer[hitPointsIndex]))
							* invSuperSampling - 0.5f;

					float u0 = getFloatRNG(seedBuffer[hitPointsIndex]);
					float u1 = getFloatRNG(seedBuffer[hitPointsIndex]);
					float u2 = getFloatRNG(seedBuffer[hitPointsIndex]);

					//						scene->camera->GenerateRay(eyePath->scrX,
					//								eyePath->scrY, width, height, &eyePath->ray,
					//								u0, u1, u2);


					ss->GenerateRay(eyePath->scrX, eyePath->scrY, width, height, &eyePath->ray, u0,
							u1, u2, &ss->camera);

					eyePath->depth = 0;
					eyePath->throughput = Spectrum(1.f, 1.f, 1.f);

					eyePath->done = false;
					eyePath->splat = false;
					eyePath->sampleIndex = hitPointsIndex;

					hitPointsIndex++;

				}
			}
		}
	}

	// Iterate through all eye paths
	//std::cerr << "Building eye paths hit points: " << std::endl;
//	lastPrintTime = WallClockTime();
	// Note: (todoEyePaths.size() > 0) is extremly slow to execute


	uint todoEyePathCount = hitPointTotal;
	uint chunk_counter = 0;

	//std::cerr << "  " << todoEyePathCount / 1000 << "k eye paths left"
	//		<< std::endl;

	uint* eyePathIndexes = new uint[getRaybufferSize()];

	while (todoEyePathCount > 0) {

//		if (WallClockTime() - lastPrintTime > 2.0) {
//			std::cerr << "  " << todoEyePathCount / 1000 << "k eye paths left" << std::endl;
//			lastPrintTime = WallClockTime();
//		}

		//std::vector<EyePath *>::iterator todoEyePathsIterator =
		//		todoEyePaths.begin() + roundPointer;

		//transversing in chunks
		uint start = (chunk_counter / getRaybufferSize()) * getRaybufferSize();

		uint end;
		if (hitPointTotal - start < getRaybufferSize())
			end = hitPointTotal;
		else
			end = start + getRaybufferSize();

		for (uint i = start; i < end; i++) {

			EyePath *eyePath = &todoEyePaths[i];

			// Check if we reached the max path depth
			if (!eyePath->done && eyePath->depth > MAX_EYE_PATH_DEPTH) {

				// Add an hit point
				//HitPointInfo &hp = *(engine->GetHitPointInfo(
				//		eyePath->pixelIndex));
				HitPointStaticInfo &hp = hitPointsStaticInfo_iterationCopy[eyePath->sampleIndex];

				hp.type = CONSTANT_COLOR;
				hp.scrX = eyePath->scrX;
				hp.scrY = eyePath->scrY;
				hp.throughput = Spectrum();

				//ihp.accumPhotonCount = 0;
				//ihp.accumReflectedFlux = Spectrum();

				//ihp.photonCount = 0;
				//hp.reflectedFlux = Spectrum();
				eyePath->done = true;

			} else if (!eyePath->done) {
				eyePath->depth++;

				uint p = RaybufferAddRay(eyePath->ray);

				eyePathIndexes[p] = i;
			}

			if (eyePath->done && !eyePath->splat) {
				--todoEyePathCount;
				chunk_counter++;
				eyePath->splat = true;
			}

			//if (rayBuffer->IsFull())
			//	break;

		}

		if (getRayBufferRayCount() > 0) {

			IntersectRayBuffer();

			//printf("%d\n",rayBuffer->GetRayCount());
			AdvanceEyePaths(&todoEyePaths[0], eyePathIndexes);

			resetRayBuffer();
		}
	}

	delete[] todoEyePaths;
	delete[] eyePathIndexes;


}

//void Worker::PushHitPoints() {
//
//	engine->lockHitPoints();
//
//	//	for (uint i = 0; i < engine->hitPointTotal; i++) {
//	//
//	//		HitPoint *hp = &(engine->hitPoints_Acc)[i];
//	//		HitPoint *ihp = GetHitPoint(i);
//	//
//	//		hp->radiance = ihp->radiance;
//	//		hp->accumRadiance = ihp->accumRadiance;
//	//		hp->photonCount = ihp->photonCount;
//	//		hp->reflectedFlux = ihp->reflectedFlux;
//	//		hp->constantHitsCount = ihp->constantHitsCount;
//	//		hp->surfaceHitsCount = ihp->surfaceHitsCount;
//	//		hp->accumPhotonRadius2 = ihp->accumPhotonRadius2;
//	//
//	//	}
//
//	memcpy(engine->hitPoints_Acc, hitPoints_iterationCopy, sizeof(HitPoint) * engine->hitPointTotal);
//
//	engine->unlockHitPoints();
//
//}
//
//void Worker::PullHitPoints() {
//
//	engine->lockHitPoints();
//
//	//		for (uint i = 0; i < engine->hitPointTotal; i++) {
//	//			HitPoint *ihp = &(engine->hitPoints)[i];
//	//			IterationHitPoint *hp = &(iterationHitPoints)[i];
//	//
//	//			hp->radiance = ihp->radiance;
//	//			hp->accumRadiance = ihp->accumRadiance;
//	//			hp->photonCount = ihp->photonCount;
//	//			hp->reflectedFlux = ihp->reflectedFlux;
//	//			hp->constantHitsCount = ihp->constantHitsCount;
//	//			hp->surfaceHitsCount = ihp->surfaceHitsCount;
//	//			hp->accumPhotonRadius2 = ihp->accumPhotonRadius2;
//	//
//	//		}
//	//
//	memcpy(hitPoints_iterationCopy, engine->hitPoints_Acc, sizeof(HitPoint) * engine->hitPointTotal);
//
//	engine->unlockHitPoints();
//
//}

void Worker::ProcessIterations(PPM* engine) {

	u_int64_t photonPerIteration = engine->photonsFirstIteration;

	uint iterationCount;

	resetRayBuffer();

	UpdateBBox();
	LookupSetHitPoints(hitPointsStaticInfo_iterationCopy, hitPoints_iterationCopy);

	uint iter = 0;
	double previousIterTime = WallClockTime();
	fprintf(stdout, "iteration, photons_iter, photons_total, photons_sec, total_time, radius, device\n");
	while (!boost::this_thread::interruption_requested() && iter < config->max_iters) {
		++iter;

		double start = WallClockTime();


		if (engine->GetIterationNumber() > MAX_ITERATIONS) {
			break;
		}

		iterationCount = engine->IncIteration();

		photonPerIteration = engine->photonsFirstIteration;

#if defined USE_SPPMPA || defined USE_SPPM
		BuildHitPoints(iterationCount);
		UpdateBBox();

#endif

#if defined USE_SPPM || defined USE_PPM
		if (iterationCount == 1)
			InitRadius(iterationCount);
#else
		InitRadius(iterationCount);
#endif

		updateDeviceHitPoints();

		ReHash(currentPhotonRadius2);//argument ignored in non-PA

		updateDeviceLookupAcc();

		photonPerIteration = AdvancePhotonPath(photonPerIteration);


		getDeviceHitpoints();

#if defined USE_PPM
		AccumulateFluxPPM(iterationCount, photonPerIteration);
#endif
#if defined USE_SPPM
		AccumulateFluxSPPM(iterationCount, photonPerIteration);
#endif
#if defined USE_SPPMPA
		AccumulateFluxSPPMPA(iterationCount, photonPerIteration);
#endif
#if defined USE_PPMPA
		AccumulateFluxPPMPA(iterationCount, photonPerIteration);
#endif

		UpdateSampleFrameBuffer(photonPerIteration);

		/**
		 * iteration lock required in PhotonTracedTotal
		 */
		engine->incPhotonTracedTotal(photonPerIteration);

		//PushHitPoints();

		profiler->additeratingTime(WallClockTime() - start);
		profiler->addIteration(1);

		if (profiler->iterationCount % 100 == 0)
			profiler->printStats(deviceID);

//		if (iterationCount % 50 == 0)
//			engine->SaveImpl(to_string<uint> (iterationCount, std::dec) + engine->fileName);


#if defined USE_SPPM || defined USE_PPM
		const float radius = hitPoints_iterationCopy[0].accumPhotonRadius2;
#else
		const float radius = currentPhotonRadius2;
#endif
		const double time = WallClockTime();
		const double totalTime = time - engine->startTime;
		const double iterTime = time - previousIterTime;
//		const float itsec = engine->GetIterationNumber() / totalTime;

		const uint photonTotal = engine->getPhotonTracedTotal();
		const float photonSec   = photonTotal / (totalTime * 1000.f);
		fprintf(stdout, "%d, %lu, %u, %f, %f, %f, %f, %d\n", iterationCount, photonPerIteration, photonTotal, photonSec, iterTime, totalTime, radius, getDeviceID());
		previousIterTime = time;

	}

}

void Worker::UpdateBBox() {

	// Calculate hit points bounding box
	//std::cerr << "Building hit points bounding box: ";

	BBox hitPointsbbox = BBox();

	for (unsigned int i = 0; i < engine->hitPointTotal; ++i) {
		HitPointStaticInfo *hp = GetHitPointInfo(i);

		if (hp->type == SURFACE)
			hitPointsbbox = Union(hitPointsbbox, hp->position);
	}

	SetBBox(hitPointsbbox);

}

#if defined USE_SPPMPA || defined USE_PPMPA
void Worker::InitRadius(uint iteration) {
#else
void Worker::InitRadius(uint /*iteration*/) {
#endif

	BBox* hitPointsbbox = GetHostBBox();

	Vector ssize = hitPointsbbox->pMax - hitPointsbbox->pMin;
	float photonRadius = ((ssize.x + ssize.y + ssize.z) / 3.f) / ((engine->width
			* engine->superSampling + engine->height * engine->superSampling) / 2.f) * 2.f;

	float photonRadius2 = photonRadius * photonRadius;

#if defined USE_SPPMPA || defined USE_PPMPA


	float g = 1;
	for (uint k = 1; k < iteration; k++)
		g *= (k + engine->alpha) / k;

	g /= iteration;

	photonRadius2 = photonRadius2 * g;
#endif

	// Expand the bounding box by used radius
	hitPointsbbox->Expand(sqrt(photonRadius2));

	// Initialize hit points field
	//const float photonRadius2 = photonRadius * photonRadius;

#if defined USE_SPPMPA || defined USE_PPMPA
	currentPhotonRadius2 = photonRadius2;
#else
	for (unsigned int i = 0; i < engine->hitPointTotal; ++i) {

		//HitPointInfo *hpinfo = engine->GetHitPointInfo(i);
		HitPoint *hp = GetHitPoint(i);

		hp->accumPhotonRadius2 = photonRadius2;

	}
#endif

}

#if defined USE_SPPM || defined USE_SPPMPA
void Worker::UpdateSampleFrameBuffer(unsigned long long iterationPhotonCount) {
#else
void Worker::UpdateSampleFrameBuffer(unsigned long long /*iterationPhotonCount*/) {
#endif

	for (unsigned int i = 0; i < engine->hitPointTotal; ++i) {
		HitPointStaticInfo *hp = GetHitPointInfo(i);
		HitPoint *ihp = GetHitPoint(i);

#if defined USE_SPPM || defined USE_SPPMPA
		const float scrX = i % engine->width;
		const float scrY = i / engine->width;

		sampleBuffer->SplatSample(scrX, scrY, ihp->radiance);
#endif

#if defined USE_PPM || defined USE_PPMPA
		sampleBuffer->SplatSample(hp->scrX, hp->scrY, ihp->radiance);
#endif

	}

	sampleFrameBuffer->Clear();

	if (sampleBuffer->GetSampleCount() > 0) {

		engine->SplatSampleBuffer(sampleFrameBuffer, true, sampleBuffer);
		sampleBuffer->Reset();
	}
}

#ifdef USE_PPM
void Worker::AccumulateFluxPPM(uint /*iteration*/, u_int64_t photonTraced) {

	photonTraced += engine->getPhotonTracedTotal();

#ifndef __DEBUG
	omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < engine->hitPointTotal; i++) {
		HitPointStaticInfo *ehp = GetHitPointInfo(i);
		HitPoint *ihp = GetHitPoint(i);

		switch (ehp->type) {
			case CONSTANT_COLOR:
			ihp->radiance = ehp->throughput;
			break;
			case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				const unsigned long long pcount = ihp->photonCount + ihp->accumPhotonCount;
				const float alpha = engine->alpha;

				const float g = alpha * pcount / (ihp->photonCount * alpha + ihp->accumPhotonCount);

				ihp->accumPhotonRadius2 *= g;

				ihp->reflectedFlux = (ihp->reflectedFlux + ihp->accumReflectedFlux) * g;

				ihp->photonCount = pcount;

				const double k = 1.0 / (M_PI * ihp->accumPhotonRadius2 * photonTraced);

				ihp->radiance = ihp->reflectedFlux * k;

				ihp->accumPhotonCount = 0;
				ihp->accumReflectedFlux = Spectrum();
			}

			break;
			default:
			assert (false);
		}

	}

	//fprintf(stderr, "Iteration %d hit point 0 reducted radius: %f\n", iteration,
	//		GetHitPoint(0)->accumPhotonRadius2);
}
#endif

#ifdef USE_SPPM
void Worker::AccumulateFluxSPPM(uint iteration, u_int64_t photonTraced) {

	photonTraced += engine->getPhotonTracedTotal();

#ifndef __DEBUG
	omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < engine->hitPointTotal; i++) {
		HitPointStaticInfo *ehp = GetHitPointInfo(i);
		HitPoint *ihp = GetHitPoint(i);

		switch (ehp->type) {
			case CONSTANT_COLOR:
			ihp->accumRadiance += ehp->throughput;
			ihp->constantHitsCount += 1;
			break;
			case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				const unsigned long long pcount = ihp->photonCount + ihp->accumPhotonCount;
				const float alpha = engine->alpha;

				const float g = alpha * pcount / (ihp->photonCount * alpha + ihp->accumPhotonCount);

				ihp->accumPhotonRadius2 *= g;

				ihp->reflectedFlux = (ihp->reflectedFlux + ihp->accumReflectedFlux) * g;

				ihp->photonCount = pcount;

				ihp->accumPhotonCount = 0;
				ihp->accumReflectedFlux = Spectrum();
			}

			ihp->surfaceHitsCount += 1;
			break;
			default:
			assert (false);
		}
		const unsigned int hitCount = ihp->constantHitsCount + ihp->surfaceHitsCount;
		if (hitCount > 0) {

			const double k = 1.0 / (M_PI * ihp->accumPhotonRadius2 * photonTraced);
			Spectrum radiance_r;
			radiance_r = (ihp->accumRadiance + ihp->surfaceHitsCount * ihp->reflectedFlux * k)
			/ hitCount;
			ihp->radiance = radiance_r;
		}
	}

	fprintf(stderr, "Iteration %d hit point 0 reducted radius: %f\n", iteration,
			GetHitPoint(0)->accumPhotonRadius2);
}
#endif

#ifdef USE_PPMPA
void Worker::AccumulateFluxPPMPA(uint iteration, u_int64_t photonTraced) {

	//photonTraced += engine->getPhotonTracedTotal();


#ifndef __DEBUG
	omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < engine->hitPointTotal; i++) {

		HitPointStaticInfo *ehp = GetHitPointInfo(i);
		HitPoint *ihp = GetHitPoint(i);

		switch (ehp->type) {
		case CONSTANT_COLOR:
			ihp->radiance = ehp->throughput;

			break;
		case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				ihp->reflectedFlux = ihp->accumReflectedFlux;

				//out of the loop
				const double k = 1.0 / (M_PI * currentPhotonRadius2 * photonTraced);

				ihp->radiance = ihp->reflectedFlux * k;

				ihp->accumPhotonCount = 0;
				ihp->accumReflectedFlux = Spectrum();
			}
			break;
		default:
			assert (false);
		}
	}
}
#endif

#ifdef USE_SPPMPA
void Worker::AccumulateFluxSPPMPA(uint iteration, u_int64_t photonTraced) {

#ifndef __DEBUG
	omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < engine->hitPointTotal; i++) {
		HitPointStaticInfo *ehp = GetHitPointInfo(i);
		HitPoint *ihp = GetHitPoint(i);

		switch (ehp->type) {
			case CONSTANT_COLOR:
			ihp->accumRadiance += ehp->throughput;
			ihp->constantHitsCount += 1;
			break;
			case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				ihp->reflectedFlux = ihp->accumReflectedFlux;
				ihp->accumPhotonCount = 0;
				ihp->accumReflectedFlux = Spectrum();
			}
			ihp->surfaceHitsCount += 1;
			break;
			default:
			assert (false);
		}

		const unsigned int hitCount = ihp->constantHitsCount + ihp->surfaceHitsCount;

		if (hitCount > 0) {

			const double k = 1.0 / (M_PI * currentPhotonRadius2 * photonTraced);

			ihp->radiance = (ihp->accumRadiance + ihp->reflectedFlux * k);

		}

	}

	for (uint i = 0; i < engine->hitPointTotal; i++) {
		HitPoint *ihp = GetHitPoint(i);

		ihp->constantHitsCount = 0;
		ihp->surfaceHitsCount = 0;
		ihp ->accumRadiance = Spectrum();
	}

	fprintf(stderr, "Iteration %d hit point 0 reducted radius: %f\n", iteration,
			currentPhotonRadius2);
}
#endif

HitPointStaticInfo *Worker::GetHitPointInfo(const unsigned int index) {

	return &(hitPointsStaticInfo_iterationCopy)[index];
}

HitPoint *Worker::GetHitPoint(const unsigned int index) {

	return &(hitPoints_iterationCopy)[index];
}

void Worker::setScene(PointerFreeScene *s) {
	ss = s;
}

uint Worker::getDeviceID() {
	return deviceID;
}
