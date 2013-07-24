/*
 * CUDA_Worker.cpp
 *
 *  Created on: Oct 31, 2012
 *      Author: rr
 */

#include "CUDA_Worker.h"
#include "omp.h"

CUDA_Worker::~CUDA_Worker() {

}

void CUDA_Worker::AdvanceEyePaths(RayBuffer *rayBuffer, EyePath* todoEyePaths, uint* eyePathIndexes) {

#ifndef __DEBUG
	omp_set_num_threads(8);
#pragma omp parallel for schedule(guided)
#endif
	for (uint i = 0; i < rayBuffer->GetRayCount(); i++) {

		EyePath *eyePath = &todoEyePaths[eyePathIndexes[i]];

		const RayHit *rayHit = &rayBuffer->GetHitBuffer()[i];

		if (rayHit->Miss()) {
			// Add an hit point
			//HitPointInfo &hp = *(engine->GetHitPointInfo(eyePath->pixelIndex));
			//HitPointPositionInfo &hp = hitPointsStaticInfo_iterationCopy[eyePath->sampleIndex];
			HitPointPositionInfo *hp= GetHitPointInfo(eyePath->sampleIndex);

			//HitPoint &hp = GetHitPoint(hitPointsIndex++);
			hp->type = CONSTANT_COLOR;
			hp->scrX = eyePath->scrX;
			hp->scrY = eyePath->scrY;

			//						if (scene->infiniteLight)
			//							hp.throughput = scene->infiniteLight->Le(
			//									eyePath->ray.d) * eyePath->throughput;
			//						else
			//							hp.throughput = Spectrum();

			if (ss->infiniteLight || ss->sunLight || ss->skyLight) {
				//	hp.throughput = scene->infiniteLight->Le(eyePath->ray.d) * eyePath->throughput;

				if (ss->infiniteLight)
					ss->InfiniteLight_Le(&hp->throughput, &eyePath->ray.d, ss->infiniteLight,
							ss->infiniteLightMap);
				if (ss->sunLight)
					ss->SunLight_Le(&hp->throughput, &eyePath->ray.d, ss->sunLight);
				if (ss->skyLight)
					ss->SkyLight_Le(&hp->throughput, &eyePath->ray.d, ss->skyLight);

				hp->throughput *= eyePath->throughput;
			} else
				hp->throughput = Spectrum();

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
				//HitPointPositionInfo* hp = hitPointsStaticInfo_iterationCopy[eyePath->sampleIndex];

				HitPointPositionInfo *hp= GetHitPointInfo(eyePath->sampleIndex);


				hp->type = CONSTANT_COLOR;
				hp->scrX = eyePath->scrX;
				hp->scrY = eyePath->scrY;
				//ihp.accumPhotonCount = 0;
				//ihp.accumReflectedFlux = Spectrum();
				//ihp.photonCount = 0;
				//hp.reflectedFlux = Spectrum();
				Vector d = -eyePath->ray.d;
				ss->AreaLight_Le(&hitPointMat->param.areaLight, &d, &N, &hp->throughput);
				hp->throughput *= eyePath->throughput;

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
					HitPointPositionInfo *hp= GetHitPointInfo(eyePath->sampleIndex);
					hp->type = CONSTANT_COLOR;
					hp->scrX = eyePath->scrX;
					hp->scrY = eyePath->scrY;
					hp->throughput = Spectrum();
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
					HitPointPositionInfo *hp= GetHitPointInfo(eyePath->sampleIndex);
					hp->type = SURFACE;
					hp->scrX = eyePath->scrX;
					hp->scrY = eyePath->scrY;
					//hp.material
					//		= (SurfaceMaterial *) triMat;
					//ihp.accumPhotonCount = 0;
					//ihp.accumReflectedFlux = Spectrum();
					//ihp.photonCount = 0;
					//hp.reflectedFlux = Spectrum();
					hp->materialSS = materialIndex;

					hp->throughput = eyePath->throughput * surfaceColor;
					hp->position = hitPoint;
					hp->wo = -eyePath->ray.d;
					hp->normal = shadeN;

					// Free the eye path
					eyePath->done = true;

					//--todoEyePathCount;
				}

			}

		}
	}

}

void CUDA_Worker::IntersectGPU(RayBuffer *rayBuffer) {

	//const double t1 = WallClockTime();

	cudaMemset(hraysBuff, 0, sizeof(RayHit) * rayBuffer->GetRayCount());
	cudaMemset(raysBuff, 0, sizeof(Ray) * rayBuffer->GetRayCount());

	double start = WallClockTime();

	cudaMemcpy(raysBuff, rayBuffer->GetRayBuffer(), sizeof(Ray) * rayBuffer->GetRayCount(),
			cudaMemcpyHostToDevice);

	intersect_wrapper(raysBuff, hraysBuff, (POINTERFREESCENE::QBVHNode*) d_qbvhBuff,
			(POINTERFREESCENE::QuadTriangle*) d_qbvhTrisBuff, rayBuffer->GetRayCount());

	cudaMemcpy(rayBuffer->GetHitBuffer(), hraysBuff, sizeof(RayHit) * rayBuffer->GetRayCount(),
			cudaMemcpyDeviceToHost);

	profiler->addRayTracingTime(WallClockTime() - start);
	profiler->addRaysTraced(rayBuffer->GetRayCount());

}

u_int64_t CUDA_Worker::AdvancePhotonPath(u_int64_t photonTarget) {

	if (!seedsBuff) {
		cudaMalloc((void**) (&seedsBuff), WORK_BUCKET_SIZE * sizeof(Seed));
		cudaMemcpy(seedsBuff, seedBuffer, WORK_BUCKET_SIZE * sizeof(Seed), cudaMemcpyHostToDevice);
	}

	photonTarget = AdvancePhotonPath_wrapper(this, engine, photonTarget);

	//cudaMemcpy(seedBuffer, seedsBuff, WORK_BUCKET_SIZE * sizeof(Seed), cudaMemcpyDeviceToHost);

	return photonTarget;
}

void CUDA_Worker::MirrorHitPoints() {
	//
	//	engine->lockHitPoints();
	//
	//	for (uint i = 0; i < engine->hitPointTotal; i++) {
	//		HitPoint *ihp = &(engine->hitPoints)[i];
	//		IterationHitPoint *hp = &(iterationHitPoints)[i];
	//
	//		hp->radiance = ihp->radiance;
	//		hp->accumRadiance = ihp->accumRadiance;
	//		hp->photonCount = ihp->photonCount;
	//		hp->reflectedFlux = ihp->reflectedFlux;
	//		hp->constantHitsCount = ihp->constantHitsCount;
	//		hp->surfaceHitsCount = ihp->surfaceHitsCount;
	//		hp->accumPhotonRadius2 = ihp->accumPhotonRadius2;
	//
	//	}
	//
	//	engine->unlockHitPoints();
	//
}

void CUDA_Worker::CommitIterationHitPoints(u_int64_t photonPerIteration) {
	//
	//	engine->lockHitPoints();
	//
	//	for (uint i = 0; i < engine->hitPointTotal; i++) {
	//		HitPoint *hp = &(engine->hitPoints)[i];
	//		IterationHitPoint *ihp = &(iterationHitPoints)[i];
	//
	//		hp->radiance = ihp->radiance;
	//		hp->accumRadiance = ihp->accumRadiance;
	//		hp->photonCount = ihp->photonCount;
	//		hp->reflectedFlux = ihp->reflectedFlux;
	//		hp->constantHitsCount = ihp->constantHitsCount;
	//		hp->surfaceHitsCount = ihp->surfaceHitsCount;
	//		hp->accumPhotonRadius2 = ihp->accumPhotonRadius2;
	//
	//	}
	//
	//	engine->incPhotonCount(photonPerIteration);
	//
	//	engine->unlockHitPoints();
	//
}

void CUDA_Worker::getDeviceHitpoints() {

	//cudaMemcpy(&engine->hitPoints[0], hitPointsBuff,
	//		sizeof(HitPoint) * engine->hitPointTotal, cudaMemcpyDeviceToHost);

	cudaMemcpy(HPsIterationRadianceFlux, workerHitPointsBuff, sizeof(HitPointRadianceFlux) * engine->hitPointTotal,
			cudaMemcpyDeviceToHost);

}

void CUDA_Worker::updateDeviceHitPoints() {

	//	checkCUDAmemory("before updateHitPoints");

	if (!workerHitPointsInfoBuff) {
		cudaMalloc((void**) (&workerHitPointsInfoBuff),
				sizeof(HitPointPositionInfo) * engine->hitPointTotal);

#if defined USE_SPPMPA || defined USE_SPPM
	}
#endif

	cudaMemcpy(workerHitPointsInfoBuff,GetHitPointInfo(0),
			sizeof(HitPointPositionInfo) * engine->hitPointTotal, cudaMemcpyHostToDevice);

#if defined USE_PPMPA || defined USE_PPM
}
#endif

	if (!workerHitPointsBuff)
		cudaMalloc((void**) (&workerHitPointsBuff), sizeof(HitPointRadianceFlux) * engine->hitPointTotal);

#if defined USE_PPMPA
	cudaMemset(workerHitPointsBuff,0, sizeof(HitPointRadianceFlux) * engine->hitPointTotal);
#else
	cudaMemcpy(workerHitPointsBuff, &HPsIterationRadianceFlux[0], sizeof(HitPointRadianceFlux) * engine->hitPointTotal,
			cudaMemcpyHostToDevice);
#endif

	checkCUDAError();

	//	checkCUDAmemory("After updateHitPoints");

}

void CUDA_Worker::CopyAcc() {

	POINTERFREESCENE::QBVHNode *nodes =
			(POINTERFREESCENE::QBVHNode *) ss->dataSet->GetAccelerator()->GetNodes();
	uint nNodes = ss->dataSet->GetAccelerator()->GetNodesCount();
	POINTERFREESCENE::QuadTriangle *prims =
			(POINTERFREESCENE::QuadTriangle *) ss->dataSet->GetAccelerator()->GetPrims();

	uint nQuads = ss->dataSet->GetAccelerator()->GetPrimsCount();

	//	qbvhBuff = new cl::Buffer(oclContext,
	//			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	//			sizeof(QBVHNode) * qbvh->nNodes, qbvh->nodes);
	//
	//	qbvhTrisBuff = new cl::Buffer(oclContext,
	//			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	//			sizeof(QuadTriangle) * qbvh->nQuads, qbvh->prims);


	//	qbvhKernel->setArg(0, *raysBuff);
	//				qbvhKernel->setArg(1, *hitsBuff);
	//				qbvhKernel->setArg(4, (unsigned int)rayBuffer->GetRayCount());
	//				oclQueue->enqueueNDRangeKernel(*qbvhKernel, cl::NullRange,
	//					cl::NDRange(rayBuffer->GetSize()), cl::NDRange(qbvhWorkGroupSize));

	cudaMalloc((void**) &d_qbvhBuff, sizeof(POINTERFREESCENE::QBVHNode) * nNodes);
	cudaMalloc((void**) &d_qbvhTrisBuff, sizeof(POINTERFREESCENE::QuadTriangle) * nQuads);

	cudaMemcpy(d_qbvhBuff, nodes, sizeof(POINTERFREESCENE::QBVHNode) * nNodes,
			cudaMemcpyHostToDevice);

	cudaMemcpy(d_qbvhTrisBuff, prims, sizeof(POINTERFREESCENE::QuadTriangle) * nQuads,
			cudaMemcpyHostToDevice);

}

void CUDA_Worker::AllocCopyCUDABuffer(void **buff, void *src, const size_t size, const string &desc) {

	cudaMalloc(buff, size);

	cudaMemcpy(*buff, src, size, cudaMemcpyHostToDevice);

	//	const OpenCLDeviceDescription *deviceDesc = intersectionDevice->GetDeviceDesc();
	//	if (*buff) {
	//		// Check the size of the already allocated buffer
	//
	//		if (size == (*buff)->getInfo<CL_MEM_SIZE>()) {
	//			// I can reuse the buffer
	//			//SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] " << desc << " buffer reused for size: " << (size / 1024) << "Kbytes");
	//			return;
	//		} else {
	//			// Free the buffer
	//			deviceDesc->FreeMemory((*buff)->getInfo<CL_MEM_SIZE>());
	//			delete *buff;
	//		}
	//	}
	//
	//	cl::Context &oclContext = intersectionDevice->GetOpenCLContext();
	//
	//	//SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] " << desc << " buffer size: " << (size / 1024) << "Kbytes");
	//	*buff = new cl::Buffer(oclContext,
	//			CL_MEM_READ_WRITE,
	//			size);
	//	deviceDesc->AllocMemory((*buff)->getInfo<CL_MEM_SIZE>());
}

void CUDA_Worker::InitCamera() {
	AllocCopyCUDABuffer((void**) &cameraBuff, &ss->camera, sizeof(POINTERFREESCENE::Camera),
			"Camera");
}

void CUDA_Worker::InitGeometry() {
	//Scene *scene = renderEngine->renderConfig->scene;
	//CompiledScene *cscene = renderEngine->compiledScene;

	const unsigned int trianglesCount = ss->dataSet->GetTotalTriangleCount();
	AllocCopyCUDABuffer((void**) &meshIDsBuff, (void *) ss->meshIDs,
			sizeof(unsigned int) * trianglesCount, "MeshIDs");

	AllocCopyCUDABuffer((void**) &normalsBuff, &ss->normals[0],
			sizeof(Normal) * ss->normals.size(), "Normals");

	AllocCopyCUDABuffer((void**) &colorsBuff, &ss->colors[0], sizeof(Spectrum) * ss->colors.size(),
			"Colors");

	if (ss->uvs.size() > 0)
		AllocCopyCUDABuffer((void**) &uvsBuff, &ss->uvs[0], sizeof(UV) * ss->uvs.size(), "UVs");
	else
		uvsBuff = NULL;

	AllocCopyCUDABuffer((void**) &vertsBuff, &ss->verts[0], sizeof(Point) * ss->verts.size(),
			"Vertices");

	AllocCopyCUDABuffer((void**) &trisBuff, &ss->tris[0], sizeof(Triangle) * ss->tris.size(),
			"Triangles");

	// Check the used accelerator type
	if (ss->dataSet->GetAcceleratorType() == ACCEL_QBVH) {
		// MQBVH geometry must be defined in a specific way.

		AllocCopyCUDABuffer((void**) &meshFirstTriangleOffsetBuff,
				(void *) ss->meshFirstTriangleOffset, sizeof(unsigned int) * ss->meshDescs.size(),
				"First mesh triangle offset");

		AllocCopyCUDABuffer((void**) &meshDescsBuff, &ss->meshDescs[0],
				sizeof(POINTERFREESCENE::Mesh) * ss->meshDescs.size(), "Mesh description");
	} else {
		meshFirstTriangleOffsetBuff = NULL;
		meshDescsBuff = NULL;
	}
}

void CUDA_Worker::InitMaterials() {
	const size_t materialsCount = ss->materials.size();
	AllocCopyCUDABuffer((void**) &materialsBuff, &ss->materials[0],
			sizeof(POINTERFREESCENE::Material) * materialsCount, "Materials");

	const unsigned int meshCount = ss->meshMats.size();
	AllocCopyCUDABuffer((void**) &meshMatsBuff, &ss->meshMats[0], sizeof(unsigned int) * meshCount,
			"Mesh material index");
}

void CUDA_Worker::InitAreaLights() {

	if (ss->areaLights.size() > 0) {
		AllocCopyCUDABuffer((void**) &areaLightsBuff, &ss->areaLights[0],
				sizeof(POINTERFREESCENE::TriangleLight) * ss->areaLights.size(), "AreaLights");
	} else
		areaLightsBuff = NULL;
}

void CUDA_Worker::InitInfiniteLight() {

	if (ss->infiniteLight) {
		AllocCopyCUDABuffer((void**) &infiniteLightBuff, ss->infiniteLight,
				sizeof(POINTERFREESCENE::InfiniteLight), "InfiniteLight");

		const unsigned int pixelCount = ss->infiniteLight->width * ss->infiniteLight->height;
		AllocCopyCUDABuffer((void**) &infiniteLightMapBuff, (void *) ss->infiniteLightMap,
				sizeof(Spectrum) * pixelCount, "InfiniteLight map");
	} else {
		infiniteLightBuff = NULL;
		infiniteLightMapBuff = NULL;
	}
}

void CUDA_Worker::InitSunLight() {

	if (ss->sunLight)
		AllocCopyCUDABuffer((void**) &sunLightBuff, ss->sunLight,
				sizeof(POINTERFREESCENE::SunLight), "SunLight");
	else
		sunLightBuff = NULL;
}

void CUDA_Worker::InitSkyLight() {

	if (ss->skyLight)
		AllocCopyCUDABuffer((void**) &skyLightBuff, ss->skyLight,
				sizeof(POINTERFREESCENE::SkyLight), "SkyLight");
	else
		skyLightBuff = NULL;
}

void CUDA_Worker::InitTextureMaps() {

	if ((ss->totRGBTexMem > 0) || (ss->totAlphaTexMem > 0)) {
		if (ss->totRGBTexMem > 0)
			AllocCopyCUDABuffer((void**) &texMapRGBBuff, ss->rgbTexMem,
					sizeof(Spectrum) * ss->totRGBTexMem, "TexMaps");
		else
			texMapRGBBuff = NULL;

		if (ss->totAlphaTexMem > 0)
			AllocCopyCUDABuffer((void**) &texMapAlphaBuff, ss->alphaTexMem,
					sizeof(float) * ss->totAlphaTexMem, "TexMaps Alpha Channel");
		else
			texMapAlphaBuff = NULL;

		AllocCopyCUDABuffer((void**) &texMapDescBuff, &ss->gpuTexMaps[0],
				sizeof(POINTERFREESCENE::TexMap) * ss->gpuTexMaps.size(), "TexMaps description");

		const unsigned int meshCount = ss->meshMats.size();
		AllocCopyCUDABuffer((void**) &meshTexsBuff, ss-> meshTexs,
				sizeof(unsigned int) * meshCount, "Mesh TexMaps index");

		if (ss->meshBumps) {
			AllocCopyCUDABuffer((void**) &meshBumpsBuff, ss->meshBumps,
					sizeof(unsigned int) * meshCount, "Mesh BumpMaps index");

			AllocCopyCUDABuffer((void**) &meshBumpsScaleBuff, ss->bumpMapScales,
					sizeof(float) * meshCount, "Mesh BuSSCENEmpMaps scales");
		} else {
			meshBumpsBuff = NULL;
			meshBumpsScaleBuff = NULL;
		}

		if (ss->meshNormalMaps)
			AllocCopyCUDABuffer((void**) &meshNormalMapsBuff, ss->meshNormalMaps,
					sizeof(unsigned int) * meshCount, "Mesh NormalMaps index");
		else
			meshNormalMapsBuff = NULL;
	} else {
		texMapRGBBuff = NULL;
		texMapAlphaBuff = NULL;
		texMapDescBuff = NULL;
		meshTexsBuff = NULL;
		meshBumpsBuff = NULL;
		meshBumpsScaleBuff = NULL;
		meshNormalMapsBuff = NULL;
	}
}

void CUDA_Worker::CopyGeometryToDevices() {

	//--------------------------------------------------------------------------
	// FrameBuffer definition
	//--------------------------------------------------------------------------

	//	InitFrameBuffer();

	//--------------------------------------------------------------------------
	// Camera definition
	//--------------------------------------------------------------------------

	InitCamera();

	//--------------------------------------------------------------------------
	// Scene geometry
	//--------------------------------------------------------------------------

	InitGeometry();

	//--------------------------------------------------------------------------
	// Translate material definitions
	//--------------------------------------------------------------------------

	InitMaterials();

	//--------------------------------------------------------------------------
	// Translate area lights
	//--------------------------------------------------------------------------

	InitAreaLights();

	//--------------------------------------------------------------------------
	// Check if there is an infinite light source
	//--------------------------------------------------------------------------

	InitInfiniteLight();

	//--------------------------------------------------------------------------
	// Check if there is an sun light source
	//--------------------------------------------------------------------------

	InitSunLight();

	//--------------------------------------------------------------------------
	// Check if there is an sky light source
	//--------------------------------------------------------------------------

	InitSkyLight();

	const unsigned int areaLightCount = ss->areaLights.size();
	if (!skyLightBuff && !sunLightBuff && !infiniteLightBuff && (areaLightCount == 0))
		throw runtime_error("There are no light sources supported by PathOCL in the scene");

	//--------------------------------------------------------------------------
	// Translate mesh texture maps
	//--------------------------------------------------------------------------

	InitTextureMaps();

	//--------------------------------------------------------------------------
	// Allocate Ray/RayHit buffers
	//--------------------------------------------------------------------------

	//const unsigned int taskCount = renderEngine->taskCount;


}

