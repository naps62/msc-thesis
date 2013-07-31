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

#include "cppbench.h"

class CUDA_Worker;



void GenerateSeedBuffer_wrapper(CUDA_Worker* worker);

void GenenerateCameraRays_wrapper(CUDA_Worker* worker);

void intersect_wrapper(Ray *rays, RayHit *rayHits,
		POINTERFREESCENE::QBVHNode *nodes,
		POINTERFREESCENE::QuadTriangle *quadTris, uint rayCount);

unsigned long long BuildPhotonMap_wrapper(CUDA_Worker* worker, Engine* engine,
		uint photontarget);

void AccumulateFluxPPM_wrapper(CUDA_Worker* worker, uint photontarget);
void AccumulateFluxSPPM_wrapper(CUDA_Worker* worker, uint photontarget);
void AccumulateFluxPPMPA_wrapper(CUDA_Worker* worker, uint photontarget);
void AccumulateFluxSPPMPA_wrapper(CUDA_Worker* worker, uint photontarget);

void SetSceneAndWorkerPointer(CUDA_Worker* worker, Engine* engine);

unsigned long long BuildHitpoints_wrapper(CUDA_Worker* worker);

void updateBBox_wrapper(CUDA_Worker* worker, BBox& bbox);

void SetNonPAInitialRadius2_wrapper(CUDA_Worker* worker, float photonRadius2);

float GetNonPAMaxRadius2_wrapper(CUDA_Worker* worker);

void UpdadeSampleBuffer_wrapper();

class CUDA_Worker: public Worker {
public:

	/**
	 * in this worker used for seed length only
	 */
	uint WORK_SIZE;

	HitPoint* workerHitPointsInfoBuff;

	lookupAcc* lookupA;

	CUDA_Worker* workerBuff;
	PointerFreeScene* ssBuff;

	unsigned long long int* photonCountBuff;
	unsigned long long* rayTraceCountBuff;


	SampleBufferElem *sampleBufferBuff;

	Seed *seedsBuff;

	EyePath* todoEyePathsBuff;

	PhotonHit* photonHitsBuff;
	unsigned long long* photonHitCountBuff;

	float* bbox_boilBuff;
	float* bbox_outBuff;

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

	CUDA_Worker() {

	}

	CUDA_Worker(uint device, Engine* engine_) {

		uint c = max(cfg->hitPointTotal, cfg->photonsFirstIteration); //One seed per sample/hitpoint path and one seed per photon path

#ifdef USE_GPU_HASH_GRID
		//lookupA = new GPUHashGrid(cfg->hitPointTotal, cfg->rebuildHash);
		lookupA = new GPUHashGrid(1 << (3 * MORTON_BITS), cfg->rebuildHash);
#endif

#ifdef USE_GPU_MORTON_HASH_GRID
		lookupA = new GPUMortonHashGrid(1 << (3 * MORTON_BITS), cfg->rebuildHash);
		//lookupA = new GPUMortonHashGrid(cfg->hitPointTotal, cfg->rebuildHash);
#endif

#ifdef USE_GPU_MORTON_GRID
		lookupA = new GPUMortonGrid(MORTON_BITS);
#endif

		WORK_SIZE = c;
		deviceID = device;
		engine = engine_;

		setScene(engine_->ss);

		workerBuff = NULL;
		seedsBuff = NULL;
		workerHitPointsInfoBuff = NULL;
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

	}

	~CUDA_Worker();

	void Start(bool buildHitPoints) {

		thread = new boost::thread(
				boost::bind(CUDA_Worker::Entry, this, buildHitPoints));

	}

	HitPoint *GetHitPoint(const unsigned int index) {

		return &(workerHitPointsInfoBuff)[index];
	}

	BBox* GetHostBBox() {
		return lookupA->getBBox();
	}

	void AdvanceEyePaths(RayBuffer *rayBuffer, EyePath *todoEyePaths,
			uint *eyePathIndexes);

	u_int64_t BuildPhotonMap(u_int64_t photontarget);

	void IntersectGPU(RayBuffer *rayBuffer);

	void UpdateBBox();

	void resetRayBuffer(bool realloc = false) {

//		if (realloc) {
//			cudaFree(raysBuff);
//			cudaFree(hraysBuff);
//
//			rayBuffer->SetSize(PHOTONMAP_WORK_BUCKET_SIZE);
//
//			cudaMalloc((void**) ((&raysBuff)),
//					sizeof(Ray) * rayBuffer->GetSize());
//			cudaMalloc((void**) ((&hraysBuff)),
//					sizeof(RayHit) * rayBuffer->GetSize());
//		}

//		rayBuffer->Reset();
//		memset(rayBuffer->GetHitBuffer(), 0,
//				sizeof(RayHit) * rayBuffer->GetSize());
//		memset(rayBuffer->GetRayBuffer(), 0,
//				sizeof(Ray) * rayBuffer->GetSize());
//
//		cudaMemset(hraysBuff, 0, sizeof(RayHit) * rayBuffer->GetSize());
//		cudaMemset(raysBuff, 0, sizeof(Ray) * rayBuffer->GetSize());

	}

	void CopyLookupAccToDevice() {

		lookupA->updateLookupTable();

	}

	float GetNonPAMaxRadius2();

	void SetNonPAInitialRadius2(float photonRadius2);

	void UpdateQueryRangeLookupAcc(uint it) {
		double start = WallClockTime();

		float currentMaxRad2 = GetCurrentMaxRadius2();

		lookupA->UpdateQueryRange(currentMaxRad2, it, GetHitPoint(0));

		double elapsed = WallClockTime() - start;

		fprintf(stderr, "Device %d: It %d Lookup update time: %.3f\n",
				getDeviceID(), it, elapsed);

	}

	void BuildLookupAcc() {

		double start = WallClockTime();

		float currentMaxRad2 = GetCurrentMaxRadius2();

		lookupA->Build(currentMaxRad2, GetHitPoint(0));

		double elapsed = WallClockTime() - start;

		//fprintf(stderr, "Device %d: Build Lookup time: %.3f\n", getDeviceID(),
		//		elapsed);

	}

	void SetBBox(BBox hitPointsbbox) {
		lookupA->setBBox(hitPointsbbox);
	}

	void getDeviceHitpoints();
	void updateDeviceHitPointsInfo(bool toHost);
	void ResetDeviceHitPointsInfo();

	void GetSampleBuffer();

	void ProcessEyePaths();

	void CommitIterationHitPoints(u_int64_t photonPerIteration);
	void MirrorHitPoints();
	void CopyAcc();

	void AccumulateFluxPPMPA(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxPPM(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxSPPM(uint iteration, u_int64_t photonTraced);
	void AccumulateFluxSPPMPA(uint iteration, u_int64_t photonTraced);

	void Render(bool buildHitPoints) {

		cudaSetDevice(deviceID);

		cudaMalloc(&rayTraceCountBuff, sizeof(unsigned long long));
		cudaMalloc((void**) (&photonCountBuff), sizeof(unsigned long long int));
		//cudaMalloc((void**) (&engineBuff), sizeof(Engine));
		cudaMalloc((void**) (&workerBuff), sizeof(CUDA_Worker));
		cudaMalloc((void**) (&ssBuff), sizeof(PointerFreeScene));
		cudaMalloc((void**) &photonHitCountBuff, sizeof(unsigned long long));
		cudaMalloc((void**) (&workerHitPointsInfoBuff),
				sizeof(HitPoint) * cfg->hitPointTotal);
		cudaMalloc((void**) (&todoEyePathsBuff),
				sizeof(EyePath) * cfg->hitPointTotal);
		cudaMalloc((void**) (&sampleBufferBuff),
				cfg->hitPointTotal * sizeof(SampleBufferElem));
		cudaMalloc((void**) (&seedsBuff), WORK_SIZE * sizeof(Seed));


		cudaMalloc((void**) &photonHitsBuff,
				sizeof(PhotonHit) * PHOTON_HIT_BUFFER_SIZE);

		uint hit_power_of_two = upper_power_of_two(cfg->hitPointTotal);
		cudaMalloc(&bbox_boilBuff, sizeof(float) * hit_power_of_two);
		int numBlocks, numThreads;
		ComputeGridSize(hit_power_of_two, 1024, numBlocks, numThreads);
		cudaMalloc(&bbox_outBuff, sizeof(float) * numBlocks);

		cudaMemset(bbox_boilBuff, 0, sizeof(float) * hit_power_of_two);
		cudaMemset(bbox_outBuff, 0, sizeof(float) * numBlocks);

		cudaMemset(workerHitPointsInfoBuff, 0,
				sizeof(HitPoint) * cfg->hitPointTotal);

		lookupA->Init();

		CopyAcc();
		CopyGeometryToDevices();

		checkCUDAError((char*) "Render");

		SetSceneAndWorkerPointer(this, engine);

		GenerateSeedBuffer_wrapper(this);

		checkCUDAmemory((char*) "Process Iterations");

		__p.reg("Process Iterations");

		engine->ProcessIterations(this, buildHitPoints);

		__p.stp("Process Iterations");

		__E(cudaFree(photonCountBuff));
		//__E(cudaFree(engineBuff));
		__E(cudaFree(ssBuff));
		__E(cudaFree(workerBuff));

	}

	static void Entry(CUDA_Worker *worker, bool buildHitPoints) {
		worker->Render(buildHitPoints);
	}

	void AllocCopyCUDABuffer(void **buff, void *src, const size_t size,
			const string & desc);
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

	//
	//	size_t RaybufferAddRay(const Ray &ray) {
	//		return rayBuffer->AddRay(ray);
	//	}
	//
	//	uint getRayBufferRayCount() {
	//		return rayBuffer->GetRayCount();
	//	}
	//
	//	void AdvanceEyePaths(EyePath* todoEyePaths, uint* eyePathIndexes) {
	//		AdvanceEyePaths(rayBuffer, todoEyePaths, eyePathIndexes);
	//	}
	//
	//	RayBuffer* GetRayBuffer() {
	//		return rayBuffer;
	//	}

	//	size_t getRaybufferSize() {
	//		return rayBuffer->GetSize();
	//	}
	//
	//	void IntersectRayBuffer() {
	//
	//		IntersectGPU(rayBuffer);
	//	}
	//

};

#endif /* CUDA_WORKER_H_ */
