/*
 * advance.cu
 *
 *  Created on: Sep 13, 2012
 *      Author: rr
 */
#include "core.h"
#include "pointerfreescene.h"
#include "hitpoints.h"
#include "renderEngine.h"
#include "cuda_utils.h"
#include "stdio.h"
#include "my_cutil_math.h"
#include "CUDA_Worker.h"
#include "cppbench.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "cudpp.h"
#include "photonmap_kernels.cuh"

void SetSceneAndWorkerPointer(CUDA_Worker* worker, Engine* engine) {

	cudaMemcpy(worker->workerBuff, worker, sizeof(CUDA_Worker),
			cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(workerBuff_c, &(worker->workerBuff), sizeof(void*));

	cudaMemcpy(worker->ssBuff, engine->ss, sizeof(PointerFreeScene),
			cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(ssBuff_c, &(worker->ssBuff), sizeof(void*));

}

void intersect_wrapper(Ray *rays, RayHit *rayHits,
		POINTERFREESCENE::QBVHNode *nodes,
		POINTERFREESCENE::QuadTriangle *quadTris, uint rayCount) {

	int sqrtn = sqrt(rayCount);

//dim3 blockDIM = dim3(16, 16);
//dim3 gridDIM = dim3((sqrtn / blockDIM.x) + 1, (sqrtn / blockDIM.y) + 1);

	dim3 blockDIM = dim3(BLOCKSIZE);
	dim3 gridDIM = dim3((rayCount / blockDIM.x) + 1);

	Intersect<<<gridDIM, blockDIM>>>(rays, rayHits, nodes, quadTris, rayCount);

	checkCUDAError("");

}

float reduction(uint hit_power_of_two, float * boil, float* out_d, uint op) {

	float* in_d = boil;
	int numBlocks1 = 0, prev_numBlocks, numThreads1;
	float*tmp;

	ComputeGridSize(hit_power_of_two, 1024, numBlocks1, numThreads1);

	int smemSize =
			(numThreads1 <= 32) ?
					2 * numThreads1 * sizeof(float) :
					numThreads1 * sizeof(float);

	reduce2<float> <<<numBlocks1, numThreads1, smemSize>>>(boil, out_d,
			hit_power_of_two, op);

	while (numBlocks1 != 1) {

		prev_numBlocks = numBlocks1;

		tmp = out_d;
		out_d = in_d;
		in_d = tmp;

		ComputeGridSize(prev_numBlocks, 1024, numBlocks1, numThreads1);

		smemSize =
				(numThreads1 <= 32) ?
						2 * numThreads1 * sizeof(float) :
						numThreads1 * sizeof(float);

		reduce2<float> <<<numBlocks1, numThreads1, smemSize>>>(in_d, out_d,
				prev_numBlocks, op);

	}

	float min;
	cudaMemcpy(&min, out_d, sizeof(float), cudaMemcpyDeviceToHost);

	return min;

}

void GenerateSeedBuffer_wrapper(CUDA_Worker* worker) {

	int numBlocks, numThreads;
	ComputeGridSize(worker->WORK_SIZE, 512, numBlocks, numThreads);
	GenerateSeedBuffer<<<numBlocks, numThreads>>>(worker->WORK_SIZE,
			worker->deviceID);

	cudaStreamSynchronize(cudaStreamDefault);

}

void GenenerateCameraRays_wrapper(CUDA_Worker* worker) {

	dim3 blockDIM;
	dim3 gridDIM;

	blockDIM = dim3(BLOCKSIZE2D, BLOCKSIZE2D);
	gridDIM = dim3(IntDivUp(cfg->width, BLOCKSIZE2D),
			IntDivUp(cfg->height, BLOCKSIZE2D));

	GenenerateCameraRays<<<gridDIM, blockDIM>>>(cfg->hitPointTotal,
			cfg->superSampling, cfg->width, cfg->height);

	cudaStreamSynchronize(cudaStreamDefault);

}

void updateBBox_wrapper(CUDA_Worker* worker, BBox& bbox) {

	uint hit_power_of_two = upper_power_of_two(cfg->hitPointTotal);

	float* boil = worker->bbox_boilBuff;
	float* out_d = worker->bbox_outBuff;

	int numBlocks, numThreads;
	ComputeGridSize(cfg->hitPointTotal, 512, numBlocks, numThreads);

	BoilPosition<HitPoint, 0> <<<numBlocks, numThreads>>>(
			worker->workerHitPointsInfoBuff, cfg->hitPointTotal, boil);
	bbox.pMin.x = reduction(hit_power_of_two, boil, out_d, 0);

	BoilPosition<HitPoint, 1> <<<numBlocks, numThreads>>>(
			worker->workerHitPointsInfoBuff, cfg->hitPointTotal, boil);
	bbox.pMin.y = reduction(hit_power_of_two, boil, out_d, 0);

	BoilPosition<HitPoint, 20> <<<numBlocks, numThreads>>>(
			worker->workerHitPointsInfoBuff, cfg->hitPointTotal, boil);
	bbox.pMin.z = reduction(hit_power_of_two, boil, out_d, 0);

	BoilPosition<HitPoint, 0> <<<numBlocks, numThreads>>>(
			worker->workerHitPointsInfoBuff, cfg->hitPointTotal, boil);
	bbox.pMax.x = reduction(hit_power_of_two, boil, out_d, 1);

	BoilPosition<HitPoint, 1> <<<numBlocks, numThreads>>>(
			worker->workerHitPointsInfoBuff, cfg->hitPointTotal, boil);
	bbox.pMax.y = reduction(hit_power_of_two, boil, out_d, 1);

	BoilPosition<HitPoint, 2> <<<numBlocks, numThreads>>>(
			worker->workerHitPointsInfoBuff, cfg->hitPointTotal, boil);
	bbox.pMax.z = reduction(hit_power_of_two, boil, out_d, 1);

//	printf("%.10f\n", bbox.pMin.x);
//	printf("%.10f\n", bbox.pMin.y);
//	printf("%.10f\n", bbox.pMin.z);
//	printf("%.10f\n", bbox.pMax.x);
//	printf("%.10f\n", bbox.pMax.y);
//	printf("%.10f\n", bbox.pMax.z);

	checkCUDAError("1");

}

void BuildHashGrid_wrapper(GPUHashGrid* lookup, HitPoint* hitpointsBuff,
		unsigned int hitPointsCount, float* BBpMin,
		float currentPhotonRadius2) {

	float cellSize = sqrtf(currentPhotonRadius2) * 2.f;
	float invCellsize = 1.0f / (cellSize);

// set hash parameters
	float3 bbMin = *((float3*) BBpMin);

	HashParams hashParams;
	hashParams.bbMin.x = bbMin.x;
	hashParams.bbMin.y = bbMin.y;
	hashParams.bbMin.z = bbMin.z;
	hashParams.eT = cfg->GetEngineType();

	hashParams.cellSize = cellSize;

	hashParams.invCellSize.x = hashParams.invCellSize.y =
			hashParams.invCellSize.z = 1.0f / cellSize;

	hashParams.SpatialHashTableSize = lookup->SpatialHashTableSize;

	cudaMemcpyToSymbol(g_Params, &hashParams, sizeof(HashParams));
// each hitpoint sphere touch at max 8 cells. cellsize based on radius
	uint maxHashes = hitPointsCount * 8;

	uint* m_HashValue;
	cudaMalloc(&m_HashValue, sizeof(uint) * maxHashes);

	int numBlocks, numThreads;
	ComputeGridSize(maxHashes, 512, numBlocks, numThreads);

	initHashValues<<<numBlocks, numThreads>>>(m_HashValue, lookup->PointIdx,
			maxHashes, lookup->SpatialHashTableSize);

	checkCUDAError("1");

	ComputeGridSize(hitPointsCount, 512, numBlocks, numThreads);

// convert each hit points cell and neighbours in hash buckets
	CalcPositionHashes<<<numBlocks, numThreads>>>(hitpointsBuff, m_HashValue,
			lookup->PointIdx, hitPointsCount, currentPhotonRadius2,
			*(Point*) BBpMin, invCellsize);

	checkCUDAError("1");

//
//	uint* H_HashValue = (uint*) malloc(sizeof(uint) * maxHashes);
//	uint* H_PointIdx = (uint*) malloc(sizeof(uint) * maxHashes);
//
//	cudaMemcpy(H_HashValue, m_HashValue, sizeof(uint) * maxHashes,
//			cudaMemcpyDeviceToHost);
//	cudaMemcpy(H_PointIdx, lookup->PointIdx, sizeof(uint) * maxHashes,
//			cudaMemcpyDeviceToHost);
//
//		for (uint i = 0; i < maxHashes; i++) {
//			printf("%u - %u\n", H_PointIdx[i], H_HashValue[i]);
//		}

	/**
	 * sort hash array by has bucket and drag hitpoints id with it
	 */
#warning TODO static
	CUDPPConfiguration config;
	config.algorithm = CUDPP_SORT_RADIX;
	config.datatype = CUDPP_UINT;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	CUDPPHandle plan;
	cudppPlan(&plan, config, maxHashes, 1, 0);
	cudppSort(plan, m_HashValue, (void*) lookup->PointIdx, 32, maxHashes);
	cudppDestroyPlan(plan);

	checkCUDAError("2");

//	printf("Sorting\n");
//
//	cudaMemcpy(H_HashValue, m_HashValue, sizeof(uint) * maxHashes,
//				cudaMemcpyDeviceToHost);
//		cudaMemcpy(H_PointIdx, lookup->PointIdx, sizeof(uint) * maxHashes,
//				cudaMemcpyDeviceToHost);
//
//			for (uint i = 0; i < maxHashes; i++) {
//				printf("%u - %u\n", H_PointIdx[i], H_HashValue[i]);
//			}

	const uint cellCount = hashParams.SpatialHashTableSize;

	ComputeGridSize(cellCount, 512, numBlocks, numThreads);

// count hitpoits per hash bucket and bucket start position in pointIdx array
	CreateHashTable<<<numBlocks, numThreads>>>(m_HashValue,
			lookup->FirstIdxBuff, lookup->NumHitpointsBuff, maxHashes,
			cellCount);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("3");

	checkCUDAmemory((char*) "After GPUHash");

//	printf("table\n");
//
//	int* H_FirstIdx = (int*) malloc(sizeof(int) * cellCount);
//	uint* H_NumPhotons = (uint*) malloc(sizeof(uint) * cellCount);
//
//	cudaMemcpy(H_FirstIdx, lookup->FirstIdxBuff, sizeof(int) * cellCount,
//			cudaMemcpyDeviceToHost);
//	cudaMemcpy(H_NumPhotons, lookup->NumHitpointsBuff, sizeof(uint) * cellCount,
//			cudaMemcpyDeviceToHost);
//
//			for (uint i = 0; i < cellCount; i++) {
//				if (H_FirstIdx[i] != -1)
//					printf("%d  - %u\n", H_FirstIdx[i],H_NumPhotons[i]);
//			}

#warning TODO mempool
	cudaFree(m_HashValue);

}

void BuildMortonHashGrid_wrapper(GPUMortonHashGrid* lookup,
		HitPoint* hitpointsBuff, unsigned int hitPointsCount, float* BBpMin,
		float currentPhotonRadius2) {

	float cellSize = sqrtf(currentPhotonRadius2) * 2.f;
	float invCellsize = 1.0f / (cellSize);

	uint* mortonIndex_d;

	cudaMalloc(&mortonIndex_d, sizeof(uint) * hitPointsCount);

// set hash parameters
	float3 bbMin = *((float3*) BBpMin);

	HashParams hashParams;
	hashParams.bbMin.x = bbMin.x;
	hashParams.bbMin.y = bbMin.y;
	hashParams.bbMin.z = bbMin.z;
	hashParams.eT = cfg->GetEngineType();

	hashParams.cellSize = cellSize;

	hashParams.invCellSize.x = hashParams.invCellSize.y =
			hashParams.invCellSize.z = 1.0f / cellSize;

	hashParams.SpatialHashTableSize = lookup->SpatialHashTableSize;

	cudaMemcpyToSymbol(g_Params, &hashParams, sizeof(HashParams));

	// generate offset lookup table
	int curEntry = 0;
	float4 offsetTable[27];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				offsetTable[curEntry].x = (i * cellSize) - cellSize;
				offsetTable[curEntry].y = (j * cellSize) - cellSize;
				offsetTable[curEntry].z = (k * cellSize) - cellSize;
				++curEntry;
			}
		}
	}

	// set the middle cell (0/0/0 offset) as the first
	offsetTable[13] = offsetTable[0];
	offsetTable[0].x = offsetTable[0].y = offsetTable[0].z = 0.0f;
	cudaMemcpyToSymbol(g_CellOffsets, offsetTable, sizeof(float4) * 27);

// each hitpoint sphere touch at max 8 cells. cellsize based on radius
	uint maxHashes = hitPointsCount;

	uint* m_HashValue;
	cudaMalloc(&m_HashValue, sizeof(uint) * maxHashes);

	int numBlocks, numThreads;
	ComputeGridSize(maxHashes, 512, numBlocks, numThreads);

	initHashValues<<<numBlocks, numThreads>>>(m_HashValue, mortonIndex_d,
			maxHashes, lookup->SpatialHashTableSize);

	checkCUDAError("1");

	Vector D;
	D.x = ceil(
			(lookup->hitPointsbbox.pMax.x - lookup->hitPointsbbox.pMin.x)
					* invCellsize);
	D.y = ceil(
			(lookup->hitPointsbbox.pMax.y - lookup->hitPointsbbox.pMin.y)
					* invCellsize);
	D.z = ceil(
			(lookup->hitPointsbbox.pMax.z - lookup->hitPointsbbox.pMin.z)
					* invCellsize);

	Point p;

	Vector totalCells = D;

	ComputeGridSize(hitPointsCount, 512, numBlocks, numThreads);

// convert each hit points cell and neighbours in hash buckets
	CalcPositionMortonHashes<<<numBlocks, numThreads>>>(hitpointsBuff,
			m_HashValue, mortonIndex_d, hitPointsCount, currentPhotonRadius2,
			*(Point*) BBpMin, invCellsize, totalCells);

	checkCUDAError("1");

//
//	uint* H_HashValue = (uint*) malloc(sizeof(uint) * maxHashes);
//	uint* H_PointIdx = (uint*) malloc(sizeof(uint) * maxHashes);
//
//	cudaMemcpy(H_HashValue, m_HashValue, sizeof(uint) * maxHashes,
//			cudaMemcpyDeviceToHost);
//	cudaMemcpy(H_PointIdx, lookup->PointIdx, sizeof(uint) * maxHashes,
//			cudaMemcpyDeviceToHost);
//
//		for (uint i = 0; i < maxHashes; i++) {
//			printf("%u - %u\n", H_PointIdx[i], H_HashValue[i]);
//		}

	/**
	 * sort hash array by has bucket and drag hitpoints id with it
	 */
#warning TODO static
	CUDPPConfiguration config;
	config.algorithm = CUDPP_SORT_RADIX;
	config.datatype = CUDPP_UINT;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	CUDPPHandle plan;
	cudppPlan(&plan, config, maxHashes, 1, 0);
	cudppSort(plan, m_HashValue, (void*) mortonIndex_d, 32, maxHashes);
	cudppDestroyPlan(plan);

	checkCUDAError("2");

//	printf("Sorting\n");
//
//	cudaMemcpy(H_HashValue, m_HashValue, sizeof(uint) * maxHashes,
//				cudaMemcpyDeviceToHost);
//		cudaMemcpy(H_PointIdx, lookup->PointIdx, sizeof(uint) * maxHashes,
//				cudaMemcpyDeviceToHost);
//
//			for (uint i = 0; i < maxHashes; i++) {
//				printf("%u - %u\n", H_PointIdx[i], H_HashValue[i]);
//			}

	HitPoint* old_order;
	cudaMalloc(&old_order, sizeof(HitPoint) * hitPointsCount);

	cudaMemcpy(old_order, hitpointsBuff, sizeof(HitPoint) * hitPointsCount,
			cudaMemcpyDeviceToDevice);

	ComputeGridSize(hitPointsCount, 512, numBlocks, numThreads);

	ReorderPoints<HitPoint> <<<numBlocks, numThreads>>>(hitpointsBuff,
			old_order, mortonIndex_d, hitPointsCount);

	const uint cellCount = hashParams.SpatialHashTableSize;

	ComputeGridSize(cellCount, 512, numBlocks, numThreads);

// count hitpoits per hash bucket and bucket start position in pointIdx array
	CreateHashTable<<<numBlocks, numThreads>>>(m_HashValue,
			lookup->FirstIdxBuff, lookup->NumHitpointsBuff, maxHashes,
			cellCount);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("3");

	checkCUDAmemory((char*) "After GPUHash");

//	printf("table\n");
//
//	int* H_FirstIdx = (int*) malloc(sizeof(int) * cellCount);
//	uint* H_NumPhotons = (uint*) malloc(sizeof(uint) * cellCount);
//
//	cudaMemcpy(H_FirstIdx, lookup->FirstIdxBuff, sizeof(int) * cellCount,
//			cudaMemcpyDeviceToHost);
//	cudaMemcpy(H_NumPhotons, lookup->NumHitpointsBuff, sizeof(uint) * cellCount,
//			cudaMemcpyDeviceToHost);
//
//			for (uint i = 0; i < cellCount; i++) {
//				if (H_FirstIdx[i] != -1)
//					printf("%d  - %u\n", H_FirstIdx[i],H_NumPhotons[i]);
//			}

#warning TODO mempool
	cudaFree(m_HashValue);
	cudaFree(mortonIndex_d);
	cudaFree(old_order);

}

void BuildMortonGrid_wrapper(GPUMortonGrid* lookup, HitPoint* hitpointsBuff,
		unsigned int hitPointsCount, float* BBpMin,
		float currentPhotonRadius2) {

	double start = WallClockTime();

	dim3 blockDIM;
	dim3 gridDIM;

	uint* mortonCodes_d;
	uint* mortonIndex_d;

	uint ncells = lookup->MortonBlockCount;

	cudaMalloc(&mortonCodes_d, sizeof(uint) * hitPointsCount);
	cudaMalloc(&mortonIndex_d, sizeof(uint) * hitPointsCount);

	HitPoint* old_order;
	cudaMalloc(&old_order, sizeof(HitPoint) * hitPointsCount);

	blockDIM = dim3(BLOCKSIZE, 1);
	gridDIM = dim3(IntDivUp(hitPointsCount, 512), 1);

	Vector invD;
	invD.x = 1 / (lookup->hitPointsbbox.pMax.x - lookup->hitPointsbbox.pMin.x);
	invD.y = 1 / (lookup->hitPointsbbox.pMax.y - lookup->hitPointsbbox.pMin.y);
	invD.z = 1 / (lookup->hitPointsbbox.pMax.z - lookup->hitPointsbbox.pMin.z);

	initHashValues<<<gridDIM, blockDIM>>>(mortonCodes_d, mortonIndex_d, ncells,
			ncells);

	GenerateHitPointsMortonCodes<<<gridDIM, blockDIM>>>(hitpointsBuff,
			hitPointsCount, mortonCodes_d, mortonIndex_d, lookup->hitPointsbbox,
			invD);

//	uint* H_HashValue = (uint*) malloc(sizeof(uint) * hitPointsCount);
//	uint* H_PointIdx = (uint*) malloc(sizeof(uint) * hitPointsCount);
//
//	cudaMemcpy(H_HashValue, mortonCodes_d, sizeof(uint) * hitPointsCount,
//			cudaMemcpyDeviceToHost);
//	cudaMemcpy(H_PointIdx, mortonIndex_d, sizeof(uint) * hitPointsCount,
//			cudaMemcpyDeviceToHost);
//
//	for (uint i = 0; i < hitPointsCount; i++) {
//		printf("%u - %u\n", H_PointIdx[i], H_HashValue[i]);
//	}
//
//	printf("\n Sorting...\n");

	CUDPPConfiguration config;
	config.algorithm = CUDPP_SORT_RADIX;
	config.datatype = CUDPP_UINT;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	CUDPPHandle plan;
	cudppPlan(&plan, config, hitPointsCount, 1, 0);
	cudppSort(plan, mortonCodes_d, (void*) mortonIndex_d, 32, hitPointsCount);
	cudppDestroyPlan(plan);

//	cudaMemcpy(H_HashValue, mortonCodes_d, sizeof(uint) * hitPointsCount,
//			cudaMemcpyDeviceToHost);
//	cudaMemcpy(H_PointIdx, mortonIndex_d, sizeof(uint) * hitPointsCount,
//			cudaMemcpyDeviceToHost);
//
//	for (uint i = 0; i < hitPointsCount; i++) {
//			printf("%u - %u\n", H_PointIdx[i], H_HashValue[i]);
//		}

	cudaMemcpy(old_order, hitpointsBuff, sizeof(HitPoint) * hitPointsCount,
			cudaMemcpyDeviceToDevice);

	ReorderPoints<HitPoint> <<<gridDIM, blockDIM>>>(hitpointsBuff, old_order,
			mortonIndex_d, hitPointsCount);

	//printf("\n Table...\n");

	int numBlocks, numThreads;
	ComputeGridSize(ncells, 512, numBlocks, numThreads);

	CreateHashTable<<<numBlocks, numThreads>>>(mortonCodes_d,
			lookup->FirstIdxBuff, lookup->NumHitpointsBuff, hitPointsCount,
			ncells);

//		int* H_FirstIdx = (int*) malloc(sizeof(int) * ncells);
//		uint* H_NumPhotons = (uint*) malloc(sizeof(uint) * ncells);
//
//		cudaMemcpy(H_FirstIdx, lookup->FirstIdxBuff, sizeof(int) * ncells,
//				cudaMemcpyDeviceToHost);
//		cudaMemcpy(H_NumPhotons, lookup->NumHitpointsBuff, sizeof(uint) * ncells,
//				cudaMemcpyDeviceToHost);
//
//		for (uint i = 0; i < ncells; i++) {
//			if (H_FirstIdx[i] != -1)
//				printf("%d  - %u\n", H_FirstIdx[i],H_NumPhotons[i]);
//		}

	cudaStreamSynchronize(cudaStreamDefault);

	cudaFree(old_order);
	cudaFree(mortonCodes_d);
	cudaFree(mortonIndex_d);

	double elapsed = WallClockTime() - start;

	printf("%f\n", elapsed);

}

void LookupHashGridKernel_wrapper(GPUHashGrid* lookup, uint phitcount,
		engineType eT, float currentPhotonRadius2) {

	__p.lsstt("Process Iterations > Iterations > Build Photon Map > Search");

	initHits<<<1, 1>>>();

	int numBlocks, numThreads;

	ComputeGridSize(phitcount, 512, numBlocks, numThreads);

	PhotonSearchHash<<<numBlocks, numThreads>>>(lookup->FirstIdxBuff,
			lookup->NumHitpointsBuff, lookup->PointIdx, currentPhotonRadius2,
			phitcount, eT, cfg->hitPointTotal);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("asdasd");

	printHits<<<1, 1>>>();

	__p.lsstp("Process Iterations > Iterations > Build Photon Map > Search");

}

void LookupMortonHashGridKernel_wrapper(GPUMortonHashGrid* lookup,
		uint phitcount, engineType eT, float currentPhotonRadius2) {

	__p.lsstt("Process Iterations > Iterations > Build Photon Map > Search");

	initHits<<<1, 1>>>();

	// find photons
	int numBlocks, numThreads;

	ComputeGridSize(phitcount, 512, numBlocks, numThreads);
	float cellSize = sqrtf(currentPhotonRadius2) * 2.f;
	float invCellsize = 1.0f / (cellSize);

	Vector D;
	D.x = ceil(
			(lookup->hitPointsbbox.pMax.x - lookup->hitPointsbbox.pMin.x)
					* invCellsize);
	D.y = ceil(
			(lookup->hitPointsbbox.pMax.y - lookup->hitPointsbbox.pMin.y)
					* invCellsize);
	D.z = ceil(
			(lookup->hitPointsbbox.pMax.z - lookup->hitPointsbbox.pMin.z)
					* invCellsize);

	Vector totalCells = D;

	PhotonSearchMortonHash<<<numBlocks, numThreads>>>(lookup->FirstIdxBuff,
			lookup->NumHitpointsBuff, currentPhotonRadius2, phitcount, eT,
			cfg->hitPointTotal, totalCells);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("asdasd");

	printHits<<<1, 1>>>();

	__p.lsstp("Process Iterations > Iterations > Build Photon Map > Search");

}

void LookupMortonGridKernel_wrapper(GPUMortonGrid* lookup, uint phitcount,
		engineType eT, float currentPhotonRadius2) {

	__p.lsstt("Process Iterations > Iterations > Build Photon Map > Search");

	initHits<<<1, 1>>>();

// find photons
	int numBlocks, numThreads;

	ComputeGridSize(phitcount, 512, numBlocks, numThreads);

	Vector invD;
	invD.x = 1 / (lookup->hitPointsbbox.pMax.x - lookup->hitPointsbbox.pMin.x);
	invD.y = 1 / (lookup->hitPointsbbox.pMax.y - lookup->hitPointsbbox.pMin.y);
	invD.z = 1 / (lookup->hitPointsbbox.pMax.z - lookup->hitPointsbbox.pMin.z);

	float cellSize = 1 / float(1 << lookup->bits_per_dim);

	float photonRadius = sqrt(currentPhotonRadius2);

	const Vector rad(photonRadius, photonRadius, photonRadius);

	Vector dist;
	dist.x = rad.x * invD.x;
	dist.y = rad.y * invD.y;
	dist.z = rad.z * invD.z;
	float dist2 = Dot(dist, dist);
	int nboxes = (int) ceil(sqrt(dist2) / cellSize);

	printf("%d\n", nboxes);

	PhotonSearchMorton<<<numBlocks, numThreads>>>(lookup->FirstIdxBuff,
			lookup->NumHitpointsBuff, currentPhotonRadius2, phitcount, eT,
			cfg->hitPointTotal, invD, lookup->hitPointsbbox.pMin, cellSize,
			nboxes);

	cudaStreamSynchronize(cudaStreamDefault);

	printHits<<<1, 1>>>();

	__p.lsstp("Process Iterations > Iterations > Build Photon Map > Search");

}

void SetNonPAInitialRadius2_wrapper(CUDA_Worker* worker, float photonRadius2) {

	int numBlocks, numThreads;
	ComputeGridSize(cfg->hitPointTotal, 512, numBlocks, numThreads);

	SetNonPAInitialRadius2<<<numBlocks, numThreads>>>(
			worker->workerHitPointsInfoBuff, cfg->hitPointTotal, photonRadius2);

	cudaStreamSynchronize(cudaStreamDefault);

}

float GetNonPAMaxRadius2_wrapper(CUDA_Worker* worker) {

	uint hit_power_of_two = upper_power_of_two(cfg->hitPointTotal);

	float* boil = worker->bbox_boilBuff;
	float* out_d = worker->bbox_outBuff;

	int numBlocks, numThreads;
	ComputeGridSize(cfg->hitPointTotal, 512, numBlocks, numThreads);

	BoilRadius2<<<numBlocks, numThreads>>>(worker->workerHitPointsInfoBuff,
			cfg->hitPointTotal, boil);

	float maxPhotonRadius2 = reduction(hit_power_of_two, boil, out_d, 1);

	return maxPhotonRadius2;

}

void SortPhotonHits(CUDA_Worker* worker, uint phitcount) {

	uint* mortonCodes_d;
	uint* mortonIndex_d;

	dim3 blockDIM;
	dim3 gridDIM;

	cudaMalloc(&mortonCodes_d, sizeof(uint) * phitcount);
	cudaMalloc(&mortonIndex_d, sizeof(uint) * phitcount);

	__p.lsstt(
				"Process Iterations > Iterations > Build Photon Map > Sort pho hits > bb");

	BBox bbox;

	uint hit_power_of_two = upper_power_of_two(phitcount);

	int numBlocks, numThreads;
	ComputeGridSize(hit_power_of_two, 1024, numBlocks, numThreads);

	float* boil;
	cudaMalloc(&boil, sizeof(float) * hit_power_of_two);

	float* out_d;
	cudaMalloc(&out_d, sizeof(float) * numBlocks);

	cudaMemset(boil, 0, sizeof(float) * hit_power_of_two);
	cudaMemset(out_d, 0, sizeof(float) * numBlocks);

	ComputeGridSize(phitcount, 512, numBlocks, numThreads);

	BoilPosition<PhotonHit, 0> <<<numBlocks, numThreads>>>(
			worker->photonHitsBuff, phitcount, boil);
	bbox.pMin.x = reduction(hit_power_of_two, boil, out_d, 0);

	BoilPosition<PhotonHit, 1> <<<numBlocks, numThreads>>>(
			worker->photonHitsBuff, phitcount, boil);
	bbox.pMin.y = reduction(hit_power_of_two, boil, out_d, 0);

	BoilPosition<PhotonHit, 2> <<<numBlocks, numThreads>>>(
			worker->photonHitsBuff, phitcount, boil);
	bbox.pMin.z = reduction(hit_power_of_two, boil, out_d, 0);

	BoilPosition<PhotonHit, 0> <<<numBlocks, numThreads>>>(
			worker->photonHitsBuff, phitcount, boil);
	bbox.pMax.x = reduction(hit_power_of_two, boil, out_d, 1);

	BoilPosition<PhotonHit, 1> <<<numBlocks, numThreads>>>(
			worker->photonHitsBuff, phitcount, boil);
	bbox.pMax.y = reduction(hit_power_of_two, boil, out_d, 1);

	BoilPosition<PhotonHit, 2> <<<numBlocks, numThreads>>>(
			worker->photonHitsBuff, phitcount, boil);
	bbox.pMax.z = reduction(hit_power_of_two, boil, out_d, 1);

//	printf("%.10f\n", bbox.pMin.x);
//	printf("%.10f\n", bbox.pMin.y);
//	printf("%.10f\n", bbox.pMin.z);
//	printf("%.10f\n", bbox.pMax.x);
//	printf("%.10f\n", bbox.pMax.y);
//	printf("%.10f\n", bbox.pMax.z);

	__p.lsstp(
					"Process Iterations > Iterations > Build Photon Map > Sort pho hits > bb");



	Vector invD;
	invD.x = 1 / (bbox.pMax.x - bbox.pMin.x);
	invD.y = 1 / (bbox.pMax.y - bbox.pMin.y);
	invD.z = 1 / (bbox.pMax.z - bbox.pMin.z);

	blockDIM = dim3(BLOCKSIZE, 1);
	gridDIM = dim3(IntDivUp(phitcount, 512), 1);
	GeneratePhotonHitMortonCodes<<<gridDIM, blockDIM>>>(phitcount,
			mortonCodes_d, mortonIndex_d, bbox, invD);

//		uint* H_PointIdx = (uint*) malloc(sizeof(uint) * phitcount);
//
//		uint* H_PointIdxpos = (uint*) malloc(sizeof(uint) * phitcount);
//
//		cudaMemcpy(H_PointIdxpos, mortonIndex_d, sizeof(uint) * phitcount,
//				cudaMemcpyDeviceToHost);
//
//		cudaMemcpy(H_PointIdx, mortonCodes_d, sizeof(uint) * phitcount,
//				cudaMemcpyDeviceToHost);
//
//		for (uint i = 0; i < phitcount; i++) {
//			printf("%u - %u\n", H_PointIdx[i], H_PointIdxpos[i]);
//		}
	/**
	 * sort hash array by has bucket and drag hitpoints id with it
	 */
	CUDPPConfiguration config;
	config.algorithm = CUDPP_SORT_RADIX;
	config.datatype = CUDPP_UINT;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	CUDPPHandle plan;
	cudppPlan(&plan, config, phitcount, 1, 0);

	__p.lsstt(
					"Process Iterations > Iterations > Build Photon Map > Sort pho hits > sort");

	cudppSort(plan, mortonCodes_d, (void*) mortonIndex_d, 32, phitcount);



	cudppDestroyPlan(plan);



	__p.lsstp(
					"Process Iterations > Iterations > Build Photon Map > Sort pho hits > sort");

	PhotonHit* old_order;
	cudaMalloc(&old_order, sizeof(PhotonHit) * phitcount);

	cudaMemcpy(old_order, worker->photonHitsBuff, sizeof(PhotonHit) * phitcount,
			cudaMemcpyDeviceToDevice);


	__p.lsstt(
				"Process Iterations > Iterations > Build Photon Map > Sort pho hits > move");

	ReorderPoints<PhotonHit> <<<gridDIM, blockDIM>>>(worker->photonHitsBuff,
			old_order, mortonIndex_d, phitcount);

	checkCUDAError("2");

	//		cudaMemcpy(H_PointIdxpos, mortonIndex_d, sizeof(uint) * phitcount,
	//				cudaMemcpyDeviceToHost);
	//
	//		cudaMemcpy(H_PointIdx, mortonCodes_d, sizeof(uint) * phitcount,
	//				cudaMemcpyDeviceToHost);
	//
	//		for (uint i = 0; i < phitcount; i++) {
	//			printf("sorted %u - %u\n", H_PointIdx[i], H_PointIdxpos[i]);
	//		}

	//
	//		/*
	//		 * Sort photon hits
	//		 */
	//		thrust::device_ptr<PhotonHit> pH(worker->photonHitsBuff);
	//		thrust::sort(pH, pH + phitcount, s_photonHitcmp());

	cudaStreamSynchronize(cudaStreamDefault);


	__p.lsstp(
						"Process Iterations > Iterations > Build Photon Map > Sort pho hits > move");


	checkCUDAmemory("asd");

	cudaFree(mortonCodes_d);
	cudaFree(mortonIndex_d);
	cudaFree(boil);
	cudaFree(out_d);
	cudaFree(old_order);

	checkCUDAmemory("asd2");


}

unsigned long long int BuildPhotonMap_wrapper(CUDA_Worker* worker,
		Engine* engine, uint slicePhotonTarget) {

	dim3 blockDIM;
	dim3 gridDIM;

	blockDIM = dim3(PHOTONPASS_MAX_THREADS_PER_BLOCK, 1);
	gridDIM = dim3((slicePhotonTarget / blockDIM.x), 1);

	unsigned long long int tracedPhotonCount = 0;
	unsigned long long photonHitCount = 0;

	cudaMemset(worker->photonCountBuff, 0, sizeof(unsigned long long int));

	double start = WallClockTime();

	cudaMemset(worker->photonHitsBuff, 0,
			sizeof(PhotonHit) * PHOTON_HIT_BUFFER_SIZE);

	cudaMemset(worker->photonHitCountBuff, 0, sizeof(unsigned long long));

	__p.lsstt("Process Iterations > Iterations > Build Photon Map > Trace");

	/*
	 * Trace photons
	 */
	fullAdvance<<<gridDIM, blockDIM>>>(slicePhotonTarget);
	cudaMemcpy(&photonHitCount, worker->photonHitCountBuff,
			sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	checkCUDAError("");

	__p.lsstp("Process Iterations > Iterations > Build Photon Map > Trace");

#ifdef SORT_PHOTONHITS

	__p.lsstt(
			"Process Iterations > Iterations > Build Photon Map > Sort pho hits");

	SortPhotonHits(worker, photonHitCount);

	__p.lsstp(
			"Process Iterations > Iterations > Build Photon Map > Sort pho hits");
#endif

	cudaMemcpy(&tracedPhotonCount, worker->photonCountBuff,
			sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	assert(slicePhotonTarget == tracedPhotonCount);

	worker->profiler->addPhotonTracingTime(WallClockTime() - start);
	worker->profiler->addPhotonsTraced(tracedPhotonCount);

	return photonHitCount;

}

void AccumulateFluxPPM_wrapper(CUDA_Worker* worker, uint photonTraced) {

	dim3 blockDIM = dim3(BLOCKSIZE, 1);
	dim3 gridDIM = dim3(IntDivUp(cfg->hitPointTotal, BLOCKSIZE));

	AccumulateFluxPPM<<<gridDIM, blockDIM>>>(photonTraced, cfg->hitPointTotal,
			cfg->alpha);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("");

}

void AccumulateFluxSPPM_wrapper(CUDA_Worker* worker, uint photonTraced) {

	dim3 blockDIM = dim3(BLOCKSIZE, 1);
	dim3 gridDIM = dim3(IntDivUp(cfg->hitPointTotal, BLOCKSIZE));

	AccumulateFluxSPPM<<<gridDIM, blockDIM>>>(photonTraced, cfg->hitPointTotal,
			cfg->alpha);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("");

}

void AccumulateFluxPPMPA_wrapper(CUDA_Worker* worker, uint photonTraced) {

	dim3 blockDIM = dim3(BLOCKSIZE, 1);
	dim3 gridDIM = dim3(IntDivUp(cfg->hitPointTotal, BLOCKSIZE));

	AccumulateFluxPPMPA<<<gridDIM, blockDIM>>>(worker->currentPhotonRadius2,
			photonTraced, cfg->hitPointTotal);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("");

}

void AccumulateFluxSPPMPA_wrapper(CUDA_Worker* worker, uint photonTraced) {

	dim3 blockDIM = dim3(BLOCKSIZE, 1);
	dim3 gridDIM = dim3(IntDivUp(cfg->hitPointTotal, BLOCKSIZE));

	AccumulateFluxSPPMPA<<<gridDIM, blockDIM>>>(worker->currentPhotonRadius2,
			photonTraced, cfg->hitPointTotal);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("");

}

unsigned long long BuildHitpoints_wrapper(CUDA_Worker* worker) {

	dim3 blockDIM;
	dim3 gridDIM;

	blockDIM = dim3(EYEPASS_MAX_THREADS_PER_BLOCK, 1);
	gridDIM = dim3((cfg->hitPointTotal / blockDIM.x) + 1, 1);

	cudaMemset(worker->rayTraceCountBuff, 0, sizeof(unsigned long long));

	fullAdvanceHitpoints<<<gridDIM, blockDIM>>>(cfg->hitPointTotal,
			worker->rayTraceCountBuff);

	unsigned long long RayTraceCount = 0;
	cudaMemcpy(&RayTraceCount, worker->rayTraceCountBuff,
			sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	checkCUDAError("0");

	return RayTraceCount;

}

void UpdadeSampleBuffer_wrapper() {

	dim3 blockDIM = dim3(BLOCKSIZE, 1);
	dim3 gridDIM = dim3(IntDivUp(cfg->hitPointTotal, BLOCKSIZE));

	HitPointToSample<<<gridDIM, blockDIM>>>(cfg->hitPointTotal);

	cudaStreamSynchronize(cudaStreamDefault);

	checkCUDAError("");

}
