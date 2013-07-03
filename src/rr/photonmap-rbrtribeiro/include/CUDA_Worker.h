/*
 * CUDA_Worker.h
 *
 *  Created on: Oct 31, 2012
 *      Author: rr
 */

#ifndef CUDA_WORKER_H_
#define CUDA_WORKER_H_

#include "core.h"
#include "hitpoints.h"
#include "List.h"
#include "cuda_utils.h"
#include "renderEngine.h"
#include "lookupAcc.h"
#include "Profiler.h"
#include "config.h"
#include "Worker.h"

class CUDA_Worker;

void intersect_wrapper(Ray *rays, RayHit *rayHits, POINTERFREESCENE::QBVHNode *nodes,
		POINTERFREESCENE::QuadTriangle *quadTris, uint rayCount);

unsigned long long AdvancePhotonPath_wrapper(CUDA_Worker* worker, PPM* engine, uint photontarget);

class CUDA_Worker: public Worker {
public:

	HitPointStaticInfo* workerHitPointsInfoBuff;
	HitPoint* workerHitPointsBuff;

	PointerFreeHashGrid* lookupA;

	RayBuffer *rayBuffer;

	size_t WORK_BUCKET_SIZE;

	Ray *raysBuff;
	RayHit *hraysBuff;
	Seed *seedsBuff;
	PhotonPath *livePhotonPathsBuff;
	POINTERFREESCENE::QBVHNode *d_qbvhBuff;
	POINTERFREESCENE::QuadTriangle *d_qbvhTrisBuff;
	//void *frameBufferBuff;
	void *alphaFrameBufferBuff;
	POINTERFREESCENE::Material *materialsBuff;
	unsigned int *meshIDsBuff;
	unsigned int *meshFirstTriangleOffsetBuff;
	POINTERFREESCENE::Mesh *meshDescsBuff;
	unsigned int *meshMatsBuff;
	POINTERFREESCENE::InfiniteLight *infiniteLightBuff;
	Spectrum *infiniteLightMapBuff;
	POINTERFREESCENE::SunLight *sunLightBuff;
	POINTERFREESCENE::SkyLight *skyLightBuff;
	Point *vertsBuff;
	Normal *normalsBuff;
	Spectrum *colorsBuff;
	Triangle *trisBuff;
	void *cameraBuff;
	POINTERFREESCENE::TriangleLight *areaLightsBuff;
	void *texMapRGBBuff;
	void *texMapAlphaBuff;
	void *texMapDescBuff;
	void *meshTexsBuff;
	void *meshBumpsBuff;
	void *meshBumpsScaleBuff;
	void *meshNormalMapsBuff;
	void *uvsBuff;

	CUDA_Worker(uint device, PointerFreeScene *ss, size_t buffer_size, Seed* sb,
			bool buildHitPoints = false) :
		Worker(sb) {

		lookupA = new PointerFreeHashGrid(engine->hitPointTotal);

		seedsBuff = NULL;
		livePhotonPathsBuff = NULL;
		raysBuff = NULL;
		hraysBuff = NULL;
		workerHitPointsInfoBuff = NULL;
		workerHitPointsBuff = NULL;

		WORK_BUCKET_SIZE = buffer_size;

		rayBuffer = new RayBuffer(buffer_size);

		setScene(ss);

		//frameBufferBuff = NULL;
		alphaFrameBufferBuff = NULL;
		materialsBuff = NULL;
		meshIDsBuff = NULL;
		meshFirstTriangleOffsetBuff = NULL;
		meshDescsBuff = NULL;
		meshMatsBuff = NULL;
		infiniteLightBuff = NULL;
		infiniteLightMapBuff = NULL;
		sunLightBuff = NULL;
		skyLightBuff = NULL;
		vertsBuff = NULL;
		normalsBuff = NULL;
		colorsBuff = NULL;
		trisBuff = NULL;
		cameraBuff = NULL;
		areaLightsBuff = NULL;
		texMapRGBBuff = NULL;
		texMapAlphaBuff = NULL;
		texMapDescBuff = NULL;
		meshTexsBuff = NULL;
		meshBumpsBuff = NULL;
		meshBumpsScaleBuff = NULL;
		meshNormalMapsBuff = NULL;
		uvsBuff = NULL;
		deviceID = device;

		thread = new std::thread(&CUDA_Worker::Entry, this, buildHitPoints);

	}

	~CUDA_Worker();


	BBox* GetHostBBox(){
			return &(lookupA->hitPointsbbox);
		}

	void AdvanceEyePaths(RayBuffer *rayBuffer, EyePath *todoEyePaths, uint *eyePathIndexes);

	u_int64_t AdvancePhotonPath(u_int64_t photontarget);

	void IntersectGPU(RayBuffer *rayBuffer);

	void resetRayBuffer() {

		rayBuffer->Reset();
		memset(rayBuffer->GetHitBuffer(), 0, sizeof(RayHit) * rayBuffer->GetSize());
		memset(rayBuffer->GetRayBuffer(), 0, sizeof(Ray) * rayBuffer->GetSize());

		cudaMemset(hraysBuff, 0, sizeof(RayHit) * rayBuffer->GetSize());
		cudaMemset(raysBuff, 0, sizeof(Ray) * rayBuffer->GetRayCount());

	}

	void updateDeviceLookupAcc() {

		lookupA->updateLookupTable();

	}

	void ReHash(float currentPhotonRadius2) {
		lookupA->ReHash(currentPhotonRadius2);
	}

	//void ProcessIterationsGPU(PPM *engine);

	size_t getRaybufferSize() {
		return rayBuffer->GetSize();
	}

	void IntersectRayBuffer() {
		IntersectGPU(rayBuffer);
	}

	void SetBBox(BBox hitPointsbbox) {
		lookupA->setBBox(hitPointsbbox);
	}

	void LookupSetHitPoints(HitPointStaticInfo* iterationHitPoints, HitPoint* workerHitPoints) {
		lookupA->setHitpoints(iterationHitPoints, workerHitPoints);
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

	RayBuffer* GetRayBuffer() {
		return rayBuffer;
	}

	void getDeviceHitpoints();
	void updateDeviceHitPoints();

	void CommitIterationHitPoints(u_int64_t photonPerIteration);
	void MirrorHitPoints();
	void CopyAcc();

	void Render(bool buildHitPoints) {

		cudaSetDevice(deviceID);
		cudaMalloc((void**) ((&raysBuff)), sizeof(Ray) * WORK_BUCKET_SIZE);
		cudaMalloc((void**) ((&hraysBuff)), sizeof(RayHit) * WORK_BUCKET_SIZE);
		CopyAcc();
		CopyGeometryToDevices();

#if defined USE_PPMPA || defined USE_PPM
		if (buildHitPoints) {
			BuildHitPoints(1);
			//PushHitPoints();

#if defined USE_PPMPA
			engine->globalHitPointsStaticInfo = hitPointsStaticInfo_iterationCopy;
#endif

		}

#if defined USE_PPMPA
		engine->waitForHitPoints->wait();
		hitPointsStaticInfo_iterationCopy = engine->globalHitPointsStaticInfo;

		//non-static info allocated and initialized in worker constructor
#endif
#endif

		ProcessIterations(engine);
	}

	static void Entry(CUDA_Worker *worker, bool buildHitPoints) {
		worker->Render(buildHitPoints);
	}

	void AllocCopyCUDABuffer(void **buff, void *src, const size_t size, const string & desc);
	void CopyGeometryToDevices();
	void InitCamera();
	void InitGeometry();
	void InitMaterials();
	void InitAreaLights();
	void InitInfiniteLight();
	void InitSunLight();
	void InitSkyLight();
	void InitTextureMaps();
	void InitKernels();

};

#endif /* CUDA_WORKER_H_ */
