/*
 * CUDA_Worker.cpp
 *
 *  Created on: Oct 31, 2012
 *      Author: rr
 */

#include "CPU_Worker.h"
#include "omp.h"

CPU_Worker::~CPU_Worker() {

}

void CPU_Worker::InitHitPoints() {

//	HPsIterationRadianceFlux = new HitPointRadianceFlux[cfg->hitPointTotal];
//	memset(HPsIterationRadianceFlux, 0,
//			sizeof(HitPointRadianceFlux) * cfg->hitPointTotal);

//if (cfg->GetEngineType() != PPMPA)
	HPsPositionInfo = new HitPoint[cfg->hitPointTotal];
	memset(HPsPositionInfo, 0, sizeof(HitPoint) * cfg->hitPointTotal);
	//else
	//	HPsPositionInfo = engine->GetHitPoints();

}

void CPU_Worker::AdvanceEyePaths(RayBuffer *rayBuffer, EyePath* todoEyePaths,
		uint* eyePathIndexes) {

#ifndef __DEBUG
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < rayBuffer->GetRayCount(); i++) {

		EyePath *eyePath = &todoEyePaths[eyePathIndexes[i]];

		const RayHit *rayHit = &rayBuffer->GetHitBuffer()[i];

		if (rayHit->Miss()) {
			// Add an hit point

			HitPoint* hp = GetHitPoint(eyePath->sampleIndex);

			hp->type = CONSTANT_COLOR;
			hp->scrX = eyePath->scrX;
			hp->scrY = eyePath->scrY;

			if (ss->infiniteLight || ss->sunLight || ss->skyLight) {

				if (ss->infiniteLight)
					ss->InfiniteLight_Le(&hp->throughput, &eyePath->ray.d,
							ss->infiniteLight, ss->infiniteLightMap);
				if (ss->sunLight)
					ss->SunLight_Le(&hp->throughput, &eyePath->ray.d,
							ss->sunLight);
				if (ss->skyLight)
					ss->SkyLight_Le(&hp->throughput, &eyePath->ray.d,
							ss->skyLight);

				hp->throughput *= eyePath->throughput;
			} else
				hp->throughput = Spectrum();

			eyePath->done = true;

		} else {

			// Something was hit
			Point hitPoint;
			Spectrum surfaceColor;
			Normal N, shadeN;

			if (engine->GetHitPointInformation(ss, &eyePath->ray, rayHit,
					hitPoint, surfaceColor, N, shadeN))
				continue;

			// Get the material
			const unsigned int currentTriangleIndex = rayHit->index;
			const unsigned int currentMeshIndex =
					ss->meshIDs[currentTriangleIndex];

			const uint materialIndex = ss->meshMats[currentMeshIndex];

			POINTERFREESCENE::Material *hitPointMat =
					&ss->materials[materialIndex];

			uint matType = hitPointMat->type;

			if (matType == MAT_AREALIGHT) {

				// Add an hit point
				HitPoint* hp = GetHitPoint(eyePath->sampleIndex);

				hp->type = CONSTANT_COLOR;
				hp->scrX = eyePath->scrX;
				hp->scrY = eyePath->scrY;

				Vector md = -eyePath->ray.d;
				ss->AreaLight_Le(&hitPointMat->param.areaLight, &md, &N,
						&hp->throughput);
				hp->throughput *= eyePath->throughput;

				// Free the eye path
				eyePath->done = true;

			} else {

				Vector wo = -eyePath->ray.d;
				float materialPdf;

				Vector wi;
				bool specularMaterial = true;
				float u0 = getFloatRNG(seedBuffer[eyePath->sampleIndex]);
				float u1 = getFloatRNG(seedBuffer[eyePath->sampleIndex]);
				float u2 = getFloatRNG(seedBuffer[eyePath->sampleIndex]);
				Spectrum f;

				switch (matType) {

				case MAT_MATTE:
					ss->Matte_Sample_f(&hitPointMat->param.matte, &wo, &wi,
							&materialPdf, &f, &shadeN, u0, u1,
							&specularMaterial);
					f *= surfaceColor;
					break;

				case MAT_MIRROR:
					ss->Mirror_Sample_f(&hitPointMat->param.mirror, &wo, &wi,
							&materialPdf, &f, &shadeN, &specularMaterial);
					f *= surfaceColor;
					break;

				case MAT_GLASS:
					ss->Glass_Sample_f(&hitPointMat->param.glass, &wo, &wi,
							&materialPdf, &f, &N, &shadeN, u0,
							&specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_MATTEMIRROR:
					ss->MatteMirror_Sample_f(&hitPointMat->param.matteMirror,
							&wo, &wi, &materialPdf, &f, &shadeN, u0, u1, u2,
							&specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_METAL:
					ss->Metal_Sample_f(&hitPointMat->param.metal, &wo, &wi,
							&materialPdf, &f, &shadeN, u0, u1,
							&specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_MATTEMETAL:
					ss->MatteMetal_Sample_f(&hitPointMat->param.matteMetal, &wo,
							&wi, &materialPdf, &f, &shadeN, u0, u1, u2,
							&specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_ALLOY:
					ss->Alloy_Sample_f(&hitPointMat->param.alloy, &wo, &wi,
							&materialPdf, &f, &shadeN, u0, u1, u2,
							&specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_ARCHGLASS:
					ss->ArchGlass_Sample_f(&hitPointMat->param.archGlass, &wo,
							&wi, &materialPdf, &f, &N, &shadeN, u0,
							&specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_NULL:
					wi = eyePath->ray.d;
					specularMaterial = 1;
					materialPdf = 1.f;

					// I have also to restore the original throughput
					//throughput = prevThroughput;
					break;

				default:
					// Huston, we have a problem...
					specularMaterial = 1;
					materialPdf = 0.f;
					break;

				}
				if ((materialPdf <= 0.f) || f.Black()) {

					// Add an hit point
					HitPoint* hp = GetHitPoint(eyePath->sampleIndex);
					hp->type = CONSTANT_COLOR;
					hp->scrX = eyePath->scrX;
					hp->scrY = eyePath->scrY;
					hp->throughput = Spectrum();

					eyePath->done = true;
				} else if (specularMaterial || (!hitPointMat->difuse)) {

					eyePath->throughput *= f / materialPdf;
					eyePath->ray = Ray(hitPoint, wi);
				} else {
					// Add an hit point
					HitPoint* hp = GetHitPoint(eyePath->sampleIndex);
					hp->type = SURFACE;
					hp->scrX = eyePath->scrX;
					hp->scrY = eyePath->scrY;

					hp->materialSS = materialIndex;

					hp->throughput = eyePath->throughput * surfaceColor;
					hp->position = hitPoint;
					hp->wo = -eyePath->ray.d;
					hp->normal = shadeN;

					// Free the eye path
					eyePath->done = true;

				}

			}

		}
	}

}

void CPU_Worker::ProcessEyePaths() {

	const unsigned int width = cfg->width;
	const unsigned int height = cfg->height;
	const unsigned int superSampling = cfg->superSampling;

	const unsigned int hitPointTotal = cfg->hitPointTotal;

	EyePath* todoEyePaths = new EyePath[hitPointTotal];

	unsigned int hitPointsIndex = 0;
	const float invSuperSampling = 1.f / superSampling;

	for (unsigned int y = 0; y < height; ++y) {
		for (unsigned int x = 0; x < width; ++x) {

			for (unsigned int sy = 0; sy < superSampling; ++sy) {
				for (unsigned int sx = 0; sx < superSampling; ++sx) {

					EyePath *eyePath = &todoEyePaths[hitPointsIndex];

					eyePath->scrX = x
							+ (sx + getFloatRNG(seedBuffer[hitPointsIndex]))
									* invSuperSampling - 0.5f;

					eyePath->scrY = y
							+ (sy + getFloatRNG(seedBuffer[hitPointsIndex]))
									* invSuperSampling - 0.5f;

					float u0 = getFloatRNG(seedBuffer[hitPointsIndex]);
					float u1 = getFloatRNG(seedBuffer[hitPointsIndex]);
					float u2 = getFloatRNG(seedBuffer[hitPointsIndex]);

					ss->GenerateRay(eyePath->scrX, eyePath->scrY, width, height,
							&eyePath->ray, u0, u1, u2, &ss->camera);

					eyePath->depth = 0;
					eyePath->throughput = Spectrum(1.f, 1.f, 1.f);

					eyePath->done = false;
					eyePath->splat = false;
					eyePath->sampleIndex = hitPointsIndex;

					HitPoint* hp = GetHitPoint(hitPointsIndex);

					hp->id = hitPointsIndex++;

				}
			}
		}
	}

	double start = WallClockTime();

	uint todoEyePathCount = cfg->hitPointTotal;
	uint chunk_counter = 0;
	unsigned long long r = 0;
	uint* eyePathIndexes = new uint[getRaybufferSize()];

	while (todoEyePathCount > 0) {

		//transversing in chunks
		uint start = (chunk_counter / getRaybufferSize()) * getRaybufferSize();

		uint end;
		if (cfg->hitPointTotal - start < getRaybufferSize())
			end = cfg->hitPointTotal;
		else
			end = start + getRaybufferSize();

		for (uint i = start; i < end; i++) {

			EyePath *eyePath = &todoEyePaths[i];

			// Check if we reached the max path depth
			if (!eyePath->done && eyePath->depth > MAX_EYE_PATH_DEPTH) {

				// Add an hit point
				HitPoint* hp = GetHitPoint(eyePath->sampleIndex);

				hp->type = CONSTANT_COLOR;
				hp->scrX = eyePath->scrX;
				hp->scrY = eyePath->scrY;
				hp->throughput = Spectrum();

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
		}

		if (getRayBufferRayCount() > 0) {

			IntersectRayBuffer();

			r += rayBuffer->GetRayCount();

			AdvanceEyePaths(&todoEyePaths[0], eyePathIndexes);

			resetRayBuffer();
		}
	}

	profiler->addRayTracingTime(WallClockTime() - start);
	profiler->addRaysTraced(r);

	delete[] eyePathIndexes;
	delete[] todoEyePaths;

}

void CPU_Worker::IntersectRay(const Ray *ray, RayHit *rayHit) {

	ss->dataSet->Intersect(ray, rayHit);

}

void CPU_Worker::Intersect(RayBuffer *rayBuffer) {

	// Trace rays
	const Ray *rb = rayBuffer->GetRayBuffer();
	RayHit *hb = rayBuffer->GetHitBuffer();

#ifndef __DEBUG
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(guided)
#endif
	for (unsigned int i = 0; i < rayBuffer->GetRayCount(); ++i) {
		hb[i].SetMiss();
		IntersectRay(&rb[i], &hb[i]);
	}

}

u_int64_t CPU_Worker::BuildPhotonMap(u_int64_t photonTarget) {

	double start = WallClockTime();

	HashGridLookup* lookup = (HashGridLookup*) lookupA;

	HitPoint* hp = GetHitPoint(0);

	uint todoPhotonCount = 0;

	PhotonPath* livePhotonPaths = new PhotonPath[GetWorkSize()];

	rayBuffer->Reset();

	size_t initc = min(rayBuffer->GetSize(), photonTarget);

	for (size_t i = 0; i < initc; ++i) {

		int p = rayBuffer->ReserveRay();

		Ray * b = &(rayBuffer->GetRayBuffer())[p];

		engine->InitPhotonPath(engine->ss, &livePhotonPaths[i], b,
				seedBuffer[i]);

	}

	while (todoPhotonCount < photonTarget) {

		Intersect(rayBuffer);

#ifndef __DEBUG
		omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(guided)
#endif
		for (unsigned int i = 0; i < rayBuffer->GetRayCount(); ++i) {

			PhotonPath *photonPath = &livePhotonPaths[i];
			Ray *ray = &rayBuffer->GetRayBuffer()[i];
			RayHit *rayHit = &rayBuffer->GetHitBuffer()[i];

			if (photonPath->done == true) {
				continue;
			}

			if (rayHit->Miss()) {
				photonPath->done = true;
			} else { // Something was hit

				Point hitPoint;
				Spectrum surfaceColor;
				Normal N, shadeN;

				if (engine->GetHitPointInformation(engine->ss, ray, rayHit,
						hitPoint, surfaceColor, N, shadeN))
					continue;

				const unsigned int currentTriangleIndex = rayHit->index;
				const unsigned int currentMeshIndex =
						engine->ss->meshIDs[currentTriangleIndex];

				POINTERFREESCENE::Material *hitPointMat =
						&engine->ss->materials[engine->ss->meshMats[currentMeshIndex]];

				uint matType = hitPointMat->type;

				if (matType == MAT_AREALIGHT) {
					photonPath->done = true;
				} else {

					float fPdf;
					Vector wi;
					Vector wo = -ray->d;
					bool specularBounce = true;

					float u0 = getFloatRNG(seedBuffer[i]);
					float u1 = getFloatRNG(seedBuffer[i]);
					float u2 = getFloatRNG(seedBuffer[i]);

					Spectrum f;

					switch (matType) {

					case MAT_MATTE:
						engine->ss->Matte_Sample_f(&hitPointMat->param.matte,
								&wo, &wi, &fPdf, &f, &shadeN, u0, u1,
								&specularBounce);

						f *= surfaceColor;
						break;

					case MAT_MIRROR:
						engine->ss->Mirror_Sample_f(&hitPointMat->param.mirror,
								&wo, &wi, &fPdf, &f, &shadeN, &specularBounce);
						f *= surfaceColor;
						break;

					case MAT_GLASS:
						engine->ss->Glass_Sample_f(&hitPointMat->param.glass,
								&wo, &wi, &fPdf, &f, &N, &shadeN, u0,
								&specularBounce);
						f *= surfaceColor;

						break;

					case MAT_MATTEMIRROR:
						engine->ss->MatteMirror_Sample_f(
								&hitPointMat->param.matteMirror, &wo, &wi,
								&fPdf, &f, &shadeN, u0, u1, u2,
								&specularBounce);
						f *= surfaceColor;

						break;

					case MAT_METAL:
						engine->ss->Metal_Sample_f(&hitPointMat->param.metal,
								&wo, &wi, &fPdf, &f, &shadeN, u0, u1,
								&specularBounce);
						f *= surfaceColor;

						break;

					case MAT_MATTEMETAL:
						engine->ss->MatteMetal_Sample_f(
								&hitPointMat->param.matteMetal, &wo, &wi, &fPdf,
								&f, &shadeN, u0, u1, u2, &specularBounce);
						f *= surfaceColor;

						break;

					case MAT_ALLOY:
						engine->ss->Alloy_Sample_f(&hitPointMat->param.alloy,
								&wo, &wi, &fPdf, &f, &shadeN, u0, u1, u2,
								&specularBounce);
						f *= surfaceColor;

						break;

					case MAT_ARCHGLASS:
						engine->ss->ArchGlass_Sample_f(
								&hitPointMat->param.archGlass, &wo, &wi, &fPdf,
								&f, &N, &shadeN, u0, &specularBounce);
						f *= surfaceColor;

						break;

					case MAT_NULL:
						wi = ray->d;
						specularBounce = 1;
						fPdf = 1.f;
						break;

					default:
						// Huston, we have a problem...
						specularBounce = 1;
						fPdf = 0.f;
						break;
					}

					if (!specularBounce) // if difuse
						lookup->AddFlux(hitPoint, shadeN, -ray->d,
								photonPath->flux, currentPhotonRadius2,
								GetHitPoint(0), ss);

					if (photonPath->depth < MAX_PHOTON_PATH_DEPTH) {
						// Build the next vertex path ray
						if ((fPdf <= 0.f) || f.Black()) {
							photonPath->done = true;
						} else {
							photonPath->depth++;
							photonPath->flux *= f / fPdf;

							// Russian Roulette
							const float p = 0.75f;
							if (photonPath->depth < 3) {
								*ray = Ray(hitPoint, wi);
							} else if (getFloatRNG(seedBuffer[i]) < p) {
								photonPath->flux /= p;
								*ray = Ray(hitPoint, wi);
							} else {
								photonPath->done = true;
							}
						}
					} else {
						photonPath->done = true;
					}
				}
			}
		}

		uint oldc = rayBuffer->GetRayCount();

		rayBuffer->Reset();

		for (unsigned int i = 0; i < oldc; ++i) {

			PhotonPath *photonPath = &livePhotonPaths[i];
			Ray *ray = &rayBuffer->GetRayBuffer()[i];

			if (photonPath->done && todoPhotonCount < photonTarget) {
				todoPhotonCount++;

				Ray n;
				engine->InitPhotonPath(engine->ss, photonPath, &n,
						seedBuffer[i]);

				livePhotonPaths[i].done = false;

				size_t p = rayBuffer->AddRay(n);
				livePhotonPaths[p] = *photonPath;

			} else if (!photonPath->done) {
				rayBuffer->AddRay(*ray);
			}
		}
	}

	profiler->addPhotonTracingTime(WallClockTime() - start);
	profiler->addPhotonsTraced(todoPhotonCount);

	rayBuffer->Reset();

	delete[] livePhotonPaths;

	float MPhotonsSec = todoPhotonCount
			/ ((WallClockTime() - start) * 1000000.f);

	fprintf(stderr, "Device %d: Photon mapping MPhotons/sec: %.3f\n", deviceID,
			MPhotonsSec);


	printf("CPU Photon contributed %u\n", lookupA->call_times);
	lookupA->call_times=0;

	return todoPhotonCount;
}

void CPU_Worker::updateDeviceHitPointsFlux() {

}

void CPU_Worker::updateDeviceHitPointsInfo(bool toHost) {
	if (toHost)
		memcpy(cfg->GetEngine()->GetHitPoints(), HPsPositionInfo,
				sizeof(HitPoint) * cfg->hitPointTotal);
	else
		memcpy(HPsPositionInfo, cfg->GetEngine()->GetHitPoints(),
				sizeof(HitPoint) * cfg->hitPointTotal);
}

//void CPU_Worker::ResetDeviceHitPointsFlux() {
//	memset(HPsIterationRadianceFlux, 0,
//			sizeof(HitPointRadianceFlux) * cfg->hitPointTotal);
//}

void CPU_Worker::ResetDeviceHitPointsInfo() {
	memset(HPsPositionInfo, 0, sizeof(HitPoint) * cfg->hitPointTotal);
}

void CPU_Worker::AccumulateFluxPPMPA(uint iteration, u_int64_t photonTraced) {

#ifndef __DEBUG
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < cfg->hitPointTotal; i++) {

		HitPoint *ihp = GetHitPoint(i);
		//HitPointRadianceFlux *ihp = GetHitPoint(i);

		ihp->radiance = Spectrum();

		switch (ihp->type) {
		case CONSTANT_COLOR:
			ihp->radiance = ihp->throughput;

			break;
		case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				ihp->reflectedFlux = ihp->accumReflectedFlux;

				//out of the loop
				const double k = 1.0
						/ (M_PI * currentPhotonRadius2 * photonTraced);

				ihp->radiance = ihp->reflectedFlux * k;

			}
			break;
		default:
			assert(false);
		}

		ihp->accumPhotonCount = 0;
		ihp->accumReflectedFlux = Spectrum();

	}
}

void CPU_Worker::AccumulateFluxPPM(uint iteration, u_int64_t photonTraced) {

	photonTraced += engine->getPhotonTracedTotal();

#ifndef __DEBUG
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < cfg->hitPointTotal; i++) {
		HitPoint *ihp = GetHitPoint(i);
		//HitPointRadianceFlux *ihp = GetHitPoint(i);

		switch (ihp->type) {
		case CONSTANT_COLOR:
			ihp->radiance = ihp->throughput;
			break;
		case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				const unsigned long long pcount = ihp->photonCount
						+ ihp->accumPhotonCount;
				const float alpha = cfg->alpha;

				const float g = alpha * pcount
						/ (ihp->photonCount * alpha + ihp->accumPhotonCount);

				ihp->accumPhotonRadius2 *= g;

				ihp->reflectedFlux = (ihp->reflectedFlux
						+ ihp->accumReflectedFlux) * g;

				ihp->photonCount = pcount;

				const double k = 1.0
						/ (M_PI * ihp->accumPhotonRadius2 * photonTraced);

				ihp->radiance = ihp->reflectedFlux * k;

				ihp->accumPhotonCount = 0;
				ihp->accumReflectedFlux = Spectrum();
			}

			break;
		default:
			assert(false);
		}

	}

	//fprintf(stderr, "Iteration %d hit point 0 reducted radius: %f\n", iteration,
	//		GetHitPoint(0)->accumPhotonRadius2);
}

void CPU_Worker::AccumulateFluxSPPM(uint iteration, u_int64_t photonTraced) {

	photonTraced += engine->getPhotonTracedTotal();

#ifndef __DEBUG
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < cfg->hitPointTotal; i++) {

		HitPoint *ihp = GetHitPoint(i);
		//HitPointRadianceFlux *ihp = GetHitPoint(i);

		switch (ihp->type) {
		case CONSTANT_COLOR:
			ihp->accumRadiance += ihp->throughput;
			ihp->constantHitsCount += 1;
			break;

		case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				const unsigned long long pcount = ihp->photonCount
						+ ihp->accumPhotonCount;
				const float alpha = cfg->alpha;

				const float g = alpha * pcount
						/ (ihp->photonCount * alpha + ihp->accumPhotonCount);

				ihp->accumPhotonRadius2 *= g;

				ihp->reflectedFlux = (ihp->reflectedFlux
						+ ihp->accumReflectedFlux) * g;

				ihp->photonCount = pcount;

				ihp->accumPhotonCount = 0;
				ihp->accumReflectedFlux = Spectrum();
			}

			ihp->surfaceHitsCount += 1;
			break;
		default:
			assert(false);
		}
		const unsigned int hitCount = ihp->constantHitsCount
				+ ihp->surfaceHitsCount;
//		if (hitCount > 0) {
//
//			const double k = 1.0
//					/ (M_PI * ihp->accumPhotonRadius2 * photonTraced);
//			Spectrum radiance_r;
//			radiance_r = (ihp->radiance
//					+ ihp->surfaceHitsCount * ihp->reflectedFlux * k)
//					/ hitCount;
//			ihp->radiance = radiance_r;
//		}

		if (hitCount > 0) {
			const double k = 1.0
					/ (M_PI * ihp->accumPhotonRadius2 * photonTraced);
			ihp->radiance = (ihp->accumRadiance
					+ ihp->surfaceHitsCount * ihp->reflectedFlux * k)
					/ hitCount;
		}

	}

	fprintf(stderr, "Iteration %d hit point 0 reducted radius: %f\n", iteration,
			GetHitPoint(0)->accumPhotonRadius2);
}

void CPU_Worker::AccumulateFluxSPPMPA(uint iteration, u_int64_t photonTraced) {

#ifndef __DEBUG
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < cfg->hitPointTotal; i++) {
		HitPoint *ihp = GetHitPoint(i);
		//HitPointRadianceFlux *ihp = GetHitPoint(i);

		switch (ihp->type) {
		case CONSTANT_COLOR:
			ihp->accumRadiance += ihp->throughput;
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
			assert(false);
		}

		const unsigned int hitCount = ihp->constantHitsCount
				+ ihp->surfaceHitsCount;

//		if (hitCount > 0) {
//
//			const double k = 1.0 / (M_PI * currentPhotonRadius2 * photonTraced);
//
//			ihp->radiance = (ihp->accumRadiance + ihp->reflectedFlux * k);
//
//		}

		if (hitCount > 0) {
			const double k = 1.0 / (M_PI * currentPhotonRadius2 * photonTraced);
			ihp->radiance = (ihp->accumRadiance
					+ ihp->surfaceHitsCount * ihp->reflectedFlux * k)
					/ hitCount;
		}

	}

//	for (uint i = 0; i < engine->hitPointTotal; i++) {
//		HitPointRadianceFlux *ihp = GetHitPoint(i);
//
//		ihp->constantHitsCount = 0;
//		ihp->surfaceHitsCount = 0;
//		ihp->accumRadiance = Spectrum();
//	}

}

void CPU_Worker::SetNonPAInitialRadius2(float photonRadius2) {

	for (unsigned int i = 0; i < cfg->hitPointTotal; ++i) {

		HitPoint *hp = GetHitPoint(i);

		hp->accumPhotonRadius2 = photonRadius2;

	}
}

float CPU_Worker::GetNonPAMaxRadius2() {
	float maxPhotonRadius2 = 0.f;
	for (unsigned int i = 0; i < cfg->hitPointTotal; ++i) {

		HitPoint *ihp = &HPsPositionInfo[i];
		//HitPointRadianceFlux *hp = &HPsIterationRadianceFlux[i];

		if (ihp->type == SURFACE)
			maxPhotonRadius2 = max(maxPhotonRadius2, ihp->accumPhotonRadius2);
	}

	return maxPhotonRadius2;
}

void CPU_Worker::GetSampleBuffer() {

	__p.lsstt(
			"Process Iterations > Iterations > Update Samples > HP to sample");

#ifndef __DEBUG
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(guided)
#endif
	for (unsigned int i = 0; i < cfg->hitPointTotal; ++i) {
		HitPoint *hp = GetHitPoint(i);
		//HitPointRadianceFlux *ihp = GetHitPoint(i);

		sampleBuffer->SplatSample(hp->scrX, hp->scrY, hp->radiance, i, hp->id);

	}

	__p.lsstp(
			"Process Iterations > Iterations > Update Samples > HP to sample");

	__p.lsstt(
			"Process Iterations > Iterations > Update Samples > Copy samples RGB");

	__p.lsstp(
			"Process Iterations > Iterations > Update Samples > Copy samples RGB");

}
