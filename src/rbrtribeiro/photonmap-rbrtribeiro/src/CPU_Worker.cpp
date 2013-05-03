/*
 * CUDA_Worker.cpp
 *
 *  Created on: Oct 31, 2012
 *      Author: rr
 */

#include "CPU_Worker.h"
#include "omp.h"

//extern uint _num_threads;
extern Config* config;

CPU_Worker::~CPU_Worker() {

}

//void CPU_Worker::myBuildHitPoints() {
//
//	const unsigned int width = engine->width;
//	const unsigned int height = engine->height;
//	const unsigned int superSampling = engine->superSampling;
//
//	const unsigned int hitPointTotal = engine->hitPointTotal;
//
//	//Seed* EyePathsSeeds = new Seed[hitPointTotal];
//
//	EyePath* todoEyePaths = new EyePath[hitPointTotal];
//
//	unsigned int hitPointsIndex = 0;
//
//	// Generate eye rays
//	//std::cerr << "Building eye paths rays with " << superSampling << "x"
//	//		<< superSampling << " super-sampling:" << std::endl;
//	//std::cerr << "  0/" << height << std::endl;
//
//	double lastPrintTime = WallClockTime();
//	const float invSuperSampling = 1.f / superSampling;
//
//	for (unsigned int y = 0; y < height; ++y) {
//
//		if (WallClockTime() - lastPrintTime > 2.0) {
//			std::cerr << "  " << y << "/" << height << std::endl;
//			lastPrintTime = WallClockTime();
//		}
//
//		//for all hitpoints
//		for (unsigned int x = 0; x < width; ++x) {
//			for (unsigned int sy = 0; sy < superSampling; ++sy) {
//				for (unsigned int sx = engine->GetIterationNumber()0; sx < superSampling; ++sx) {
//
//					EyePath *eyePath = &todoEyePaths[hitPointsIndex];
//
//					//EyePathsSeeds[hitPointsIndex] = mwc(hitPointsIndex + deviceID);
//
//				//	Seed s = seedBuffer[hitPointsIndex];
//
//					eyePath->scrX = x + (sx + getFloatRNG(seedBuffer[hitPointsIndex]))
//							* invSuperSampling - 0.5f;
//
//					eyePath->scrY = y + (sy + getFloatRNG(seedBuffer[hitPointsIndex]))
//							* invSuperSampling - 0.5f;
//
//					float u0 = getFloatRNG(seedBuffer[hitPointsIndex]);
//					float u1 = getFloatRNG(seedBuffer[hitPointsIndex]);
//					float u2 = getFloatRNG(seedBuffer[hitPointsIndex]);
//
//					//						scene->camera->GenerateRay(eyePath->scrX,
//					//								eyePath->scrY, width, height, &eyePath->ray,
//					//								u0, u1, u2);
//
//
//					ss->GenerateRay(eyePath->scrX, eyePath->scrY, width, height, &eyePath->ray, u0,
//							u1, u2, &ss->camera);
//
//					eyePath->depth = 0;
//					eyePath->throughput = Spectrum(1.f, 1.f, 1.f);
//
//					eyePath->done = false;
//					eyePath->splat = false;
//					eyePath->sampleIndex = hitPointsIndex;
//
//					hitPointsIndex++;
//
//				}
//			}
//		}
//	}
//
//	// Iterate through all eye paths
//	//std::cerr << "Building eye paths hit points: " << std::endl;
//	lastPrintTime = WallClockTime();
//	// Note: (todoEyePaths.size() > 0) is extremly slow to execute
//
//
//	uint todoEyePathCount = hitPointTotal;
//	uint chunk_counter = 0;
//
//	//std::cerr << "  " << todoEyePathCount / 1000 << "k eye paths left"
//	//		<< std::endl;
//
//	uint c = min((uint) rayBuffer->GetSize(), todoEyePathCount);
//
//	uint* eyePathIndexes = new uint[c];
//
//	while (todoEyePathCount > 0) {
//
//		if (WallClockTime() - lastPrintTime > 2.0) {
//			std::cerr << "  " << todoEyePathCount / 1000 << "k eye paths left" << std::endl;
//			lastPrintTime = WallClockTime();
//		}
//
//		//std::vector<EyePath *>::iterator todoEyePathsIterator =
//		//		todoEyePaths.begin() + roundPointer;
//
//		//transversing in chunks
//		uint start = (chunk_counter / rayBuffer->GetSize()) * rayBuffer->GetSize();
//
//		uint end;
//		if (hitPointTotal - start < rayBuffer->GetSize())
//			end = hitPointTotal;
//		else
//			end = start + rayBuffer->GetSize();
//
//		for (uint i = start; i < end; i++) {
//
//			EyePath *eyePath = &todoEyePaths[i];
//
//			// Check if we reached the max path depth
//			if (!eyePath->done && eyePath->depth > MAX_EYE_PATH_DEPTH) {
//
//				// Add an hit point
//				//HitPointInfo &hp = *(engine->GetHitPointInfo(
//				//		eyePath->pixelIndex));
//				IterationHitPoint &hp = iterationHitPoints[eyePath->sampleIndex];
//
//				hp.type = CONSTANT_COLOR;
//				hp.scrX = eyePath->scrX;
//				hp.scrY = eyePath->scrY;
//				hp.throughput = Spectrum();
//
//				//ihp.accumPhotonCount = 0;
//				//ihp.accumReflectedFlux = Spectrum();
//
//				//ihp.photonCount = 0;
//				//hp.reflectedFlux = Spectrum();
//				eyePath->done = true;
//
//			} else if (!eyePath->done) {
//				eyePath->depth++;
//				eyePathIndexes[rayBuffer->AddRay(eyePath->ray)] = i;
//
//			}
//
//			if (eyePath->done && !eyePath->splat) {
//				--todoEyePathCount;
//				chunk_counter++;
//				eyePath->splat = true;
//			}
//
//			//if (rayBuffer->IsFull())
//			//	break;
//
//		}
//
//		if (rayBuffer->GetRayCount() > 0) {
//
//			Intersect(rayBuffer);
//
//			//printf("%d\n",rayBuffer->GetRayCount());
//			AdvanceEyePaths(rayBuffer, &todoEyePaths[0], eyePathIndexes);
//
//			rayBuffer->Reset();
//		}
//	}
//
//
//	delete[] todoEyePaths;
//	delete[] eyePathIndexes;
//
//	// Calculate hit points bounding box
//	//std::cerr << "Building hit points bounding box: ";
//	for (unsigned int i = 0; i < engine->hitPointTotal; ++i) {
//		IterationHitPoint *hp = GetHitPoint(i);
//
//		if (hp->type == SURFACE)
//			hitPointsbbox = Union(hitPointsbbox, hp->position);
//	}
//
//	lookupA->setBBox(hitPointsbbox);
//	lookupA->setHitpoints(iterationHitPoints);
//
//}

void CPU_Worker::AdvanceEyePaths( RayBuffer *rayBuffer, EyePath* todoEyePaths,
		uint* eyePathIndexes) {

	const uint max = rayBuffer->GetRayCount();

omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
	for (uint i = 0; i < max; i++) {

		EyePath *eyePath = &todoEyePaths[eyePathIndexes[i]];

		const RayHit *rayHit = &rayBuffer->GetHitBuffer()[i];

		if (rayHit->Miss()) {
			// Add an hit point
			//HitPointInfo &hp = *(engine->GetHitPointInfo(eyePath->pixelIndex));
			HitPointStaticInfo &hp = hitPointsStaticInfo_iterationCopy[eyePath->sampleIndex];

			//HitPoint &hp = GetHitPoint(hitPointsIndex++);
			hp.type = CONSTANT_COLOR;
			hp.scrX = eyePath->scrX;
			hp.scrY = eyePath->scrY;

			//						if (scene->infiniteLight)
			//							hp.throughput = scene->infiniteLight->Le(
			//									eyePath->ray.d) * eyePath->throughput;
			//						else
			//							hp.throughput = Spectrum();

			if (ss->infiniteLight || ss->sunLight || ss->skyLight) {
				//	hp.throughput = scene->infiniteLight->Le(eyePath->ray.d) * eyePath->throughput;

				if (ss->infiniteLight)
					ss->InfiniteLight_Le(&hp.throughput, &eyePath->ray.d, ss->infiniteLight,
							ss->infiniteLightMap);
				if (ss->sunLight)
					ss->SunLight_Le(&hp.throughput, &eyePath->ray.d, ss->sunLight);
				if (ss->skyLight)
					ss->SkyLight_Le(&hp.throughput, &eyePath->ray.d, ss->skyLight);

				hp.throughput *= eyePath->throughput;
			} else
				hp.throughput = Spectrum();

			// Free the eye path
			//ihp.accumPhotonCount = 0;
			//ihp.accumReflectedFlux = Spectrum();
			//ihp.photonCount = 0;
			//hp.reflectedFlux = Spectrum();
			eyePath->done = true;

			//--todoEyePathCount;

		} else {

			// Something was hit
			Point hitPoint;
			Spectrum surfaceColor;
			Normal N, shadeN;

			if (engine->GetHitPointInformation(ss, &eyePath->ray, rayHit, hitPoint, surfaceColor,
					N, shadeN))
				continue;

			// Get the material
			const unsigned int currentTriangleIndex = rayHit->index;
			const unsigned int currentMeshIndex = ss->meshIDs[currentTriangleIndex];

			const uint materialIndex = ss->meshMats[currentMeshIndex];

			POINTERFREESCENE::Material *hitPointMat = &ss->materials[materialIndex];

			uint matType = hitPointMat->type;

			if (matType == MAT_AREALIGHT) {

				// Add an hit point
				//HitPointInfo &hp = *(engine->GetHitPointInfo(
				//		eyePath->pixelIndex));
				HitPointStaticInfo &hp = hitPointsStaticInfo_iterationCopy[eyePath->sampleIndex];

				hp.type = CONSTANT_COLOR;
				hp.scrX = eyePath->scrX;
				hp.scrY = eyePath->scrY;
				//ihp.accumPhotonCount = 0;
				//ihp.accumReflectedFlux = Spectrum();
				//ihp.photonCount = 0;
				//hp.reflectedFlux = Spectrum();

				Vector md = -eyePath->ray.d;
				ss->AreaLight_Le(&hitPointMat->param.areaLight, &md, &N,
						&hp.throughput);
				hp.throughput *= eyePath->throughput;

				// Free the eye path
				eyePath->done = true;

				//--todoEyePathCount;

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
					ss->Matte_Sample_f(&hitPointMat->param.matte, &wo, &wi, &materialPdf, &f,
							&shadeN, u0, u1, &specularMaterial);
					f *= surfaceColor;
					break;

				case MAT_MIRROR:
					ss->Mirror_Sample_f(&hitPointMat->param.mirror, &wo, &wi, &materialPdf, &f,
							&shadeN, &specularMaterial);
					f *= surfaceColor;
					break;

				case MAT_GLASS:
					ss->Glass_Sample_f(&hitPointMat->param.glass, &wo, &wi, &materialPdf, &f, &N,
							&shadeN, u0, &specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_MATTEMIRROR:
					ss->MatteMirror_Sample_f(&hitPointMat->param.matteMirror, &wo, &wi,
							&materialPdf, &f, &shadeN, u0, u1, u2, &specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_METAL:
					ss->Metal_Sample_f(&hitPointMat->param.metal, &wo, &wi, &materialPdf, &f,
							&shadeN, u0, u1, &specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_MATTEMETAL:
					ss->MatteMetal_Sample_f(&hitPointMat->param.matteMetal, &wo, &wi, &materialPdf,
							&f, &shadeN, u0, u1, u2, &specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_ALLOY:
					ss->Alloy_Sample_f(&hitPointMat->param.alloy, &wo, &wi, &materialPdf, &f,
							&shadeN, u0, u1, u2, &specularMaterial);
					f *= surfaceColor;

					break;

				case MAT_ARCHGLASS:
					ss->ArchGlass_Sample_f(&hitPointMat->param.archGlass, &wo, &wi, &materialPdf,
							&f, &N, &shadeN, u0, &specularMaterial);
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

				//						if (f.r != f2.r || f.g != f2.g || f.b != f2.b) {
				//							printf("d");
				//						}

				if ((materialPdf <= 0.f) || f.Black()) {

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
					// Free the eye path
					eyePath->done = true;

					//--todoEyePathCount;
				} else if (specularMaterial || (!hitPointMat->difuse)) {

					eyePath->throughput *= f / materialPdf;
					eyePath->ray = Ray(hitPoint, wi);
				} else {
					// Add an hit point
					//HitPointInfo &hp = *(engine->GetHitPointInfo(
					//		eyePath->pixelIndex));
					HitPointStaticInfo &hp = hitPointsStaticInfo_iterationCopy[eyePath->sampleIndex];
					hp.type = SURFACE;
					hp.scrX = eyePath->scrX;
					hp.scrY = eyePath->scrY;
					//hp.material
					//		= (SurfaceMaterial *) triMat;
					//ihp.accumPhotonCount = 0;
					//ihp.accumReflectedFlux = Spectrum();
					//ihp.photonCount = 0;
					//hp.reflectedFlux = Spectrum();
					hp.materialSS = materialIndex;

					hp.throughput = eyePath->throughput * surfaceColor;
					hp.position = hitPoint;
					hp.wo = -eyePath->ray.d;
					hp.normal = shadeN;

					// Free the eye path
					eyePath->done = true;

					//--todoEyePathCount;
				}

			}

		}
	}

}

void CPU_Worker::IntersectRay(const Ray *ray, RayHit *rayHit) {

	ss->dataSet->Intersect(ray, rayHit);

}

void CPU_Worker::Intersect(RayBuffer *rayBuffer) {

	// Trace rays
	const Ray *rb = rayBuffer->GetRayBuffer();
	RayHit *hb = rayBuffer->GetHitBuffer();

	double start = WallClockTime();

#ifndef __DEBUG
	omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
	for (unsigned int i = 0; i < rayBuffer->GetRayCount(); ++i) {
		hb[i].SetMiss();
		IntersectRay(&rb[i], &hb[i]);
	}

	profiler->addRayTracingTime(WallClockTime() - start);
	profiler->addRaysTraced(rayBuffer->GetSize());

}

u_int64_t CPU_Worker::AdvancePhotonPath(u_int64_t photonTarget) {



	uint todoPhotonCount = 0;

	PhotonPath* livePhotonPaths = new PhotonPath[rayBuffer->GetSize()];

	rayBuffer->Reset();

	size_t initc = min((int) rayBuffer->GetSize(), (int) photonTarget);

	double start = WallClockTime();


	for (size_t i = 0; i < initc; ++i) {

		int p = rayBuffer->ReserveRay();

		Ray * b = &(rayBuffer->GetRayBuffer())[p];

		engine->InitPhotonPath(engine->ss, &livePhotonPaths[i], b, seedBuffer[i]);

	}


	while (todoPhotonCount < photonTarget) {

		Intersect(rayBuffer);

#ifndef __DEBUG
		omp_set_num_threads(config->max_threads);
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

				if (engine->GetHitPointInformation(engine->ss, ray, rayHit, hitPoint, surfaceColor,
						N, shadeN))
					continue;

				const unsigned int currentTriangleIndex = rayHit->index;
				const unsigned int currentMeshIndex = engine->ss->meshIDs[currentTriangleIndex];

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
						engine->ss->Matte_Sample_f(&hitPointMat->param.matte, &wo, &wi, &fPdf, &f,
								&shadeN, u0, u1, &specularBounce);

						f *= surfaceColor;
						break;

					case MAT_MIRROR:
						engine->ss->Mirror_Sample_f(&hitPointMat->param.mirror, &wo, &wi, &fPdf,
								&f, &shadeN, &specularBounce);
						f *= surfaceColor;
						break;

					case MAT_GLASS:
						engine->ss->Glass_Sample_f(&hitPointMat->param.glass, &wo, &wi, &fPdf, &f,
								&N, &shadeN, u0, &specularBounce);
						f *= surfaceColor;

						break;

					case MAT_MATTEMIRROR:
						engine->ss->MatteMirror_Sample_f(&hitPointMat->param.matteMirror, &wo, &wi,
								&fPdf, &f, &shadeN, u0, u1, u2, &specularBounce);
						f *= surfaceColor;

						break;

					case MAT_METAL:
						engine->ss->Metal_Sample_f(&hitPointMat->param.metal, &wo, &wi, &fPdf, &f,
								&shadeN, u0, u1, &specularBounce);
						f *= surfaceColor;

						break;

					case MAT_MATTEMETAL:
						engine->ss->MatteMetal_Sample_f(&hitPointMat->param.matteMetal, &wo, &wi,
								&fPdf, &f, &shadeN, u0, u1, u2, &specularBounce);
						f *= surfaceColor;

						break;

					case MAT_ALLOY:
						engine->ss->Alloy_Sample_f(&hitPointMat->param.alloy, &wo, &wi, &fPdf, &f,
								&shadeN, u0, u1, u2, &specularBounce);
						f *= surfaceColor;

						break;

					case MAT_ARCHGLASS:
						engine->ss->ArchGlass_Sample_f(&hitPointMat->param.archGlass, &wo, &wi,
								&fPdf, &f, &N, &shadeN, u0, &specularBounce);
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
						lookupA->AddFlux(engine->ss, engine->alpha, hitPoint, shadeN, -ray->d,
								photonPath->flux, currentPhotonRadius2);

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
				engine->InitPhotonPath(engine->ss, photonPath, &n, seedBuffer[i]);

				livePhotonPaths[i].done = false;

				size_t p = rayBuffer->AddRay(n);
				livePhotonPaths[p] = *photonPath;

			} else if (!photonPath->done) {
				rayBuffer->AddRay(*ray);
			}
		}
	}


//	float MPhotonsSec = todoPhotonCount / ((WallClockTime()-start) * 1000000.f);

	//printf("\nRate: %.3f MPhotons/sec\n",MPhotonsSec);


	profiler->addPhotonTracingTime(WallClockTime() - start);
	profiler->addPhotonsTraced(todoPhotonCount);

	rayBuffer->Reset();

	return todoPhotonCount;
}


void CPU_Worker::updateDeviceHitPoints() {



}

//void CPU_Worker::InitFrameBuffer() {
//	//--------------------------------------------------------------------------
//	// FrameBuffer definition
//	//--------------------------------------------------------------------------
//
//	ss->frameBufferPixelCount = (engine->width - +2) * (engine->height + 2);
//
//	// Delete previous allocated frameBuffer
//	delete[] ss->frameBuffer;
//	ss->frameBuffer = new CUDASCENE::Pixel[ss->frameBufferPixelCount];
//
//	for (unsigned int i = 0; i < ss->frameBufferPixelCount; ++i) {
//		ss->frameBuffer[i].c.r = 0.f;
//		ss->frameBuffer[i].c.g = 0.f;
//		ss->frameBuffer[i].c.b = 0.f;
//		ss->frameBuffer[i].count = 0.f;
//	}
//
//	delete[] ss->alphaFrameBuffer;
//	ss->alphaFrameBuffer = NULL;
//
//}
