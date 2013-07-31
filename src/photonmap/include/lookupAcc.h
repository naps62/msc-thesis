/*
 * lookupAcc.h
 *
 *  Created on: Nov 9, 2012
 *      Author: rr
 */

#ifndef LOOKUPACC_H_
#define LOOKUPACC_H_

#include "renderEngine.h"
#include "core.h"
#include "cuda_utils.h"
#include "RenderConfig.h"
#include "math.h"

enum lookupAccType {
	HASH_GRID, KD_TREE
};

/**
 * At least one hash bucket per grid cell must be assured.
 * Since the cell size is based on the initial radius, which is based on the hitpoints
 * if hashsize >= hitpoints it works.
 */
class lookupAcc {
public:
	lookupAcc();
	virtual ~lookupAcc();

	virtual void setBBox(BBox d)=0;
	virtual BBox* getBBox()=0;

	virtual void UpdateQueryRange(float currentPhotonRadius2, uint it,
			HitPoint *workerHitPointsInfo)=0;

	virtual void Build(float currentPhotonRadius2,
			HitPoint *workerHitPointsInfo)=0;

	virtual void updateLookupTable() {
	}

	virtual void LookupPhotonHits(unsigned long long photonHitCount,
			float currentPhotonRadius2) {

	}

	virtual void Init(){

	}

	uint call_times;
};

/**
 * CPU linked lists hash function hash grid with neighbour pre-processing
 * cell size based on bbox and image resolution
 */
class HashGridLookup: public lookupAcc {
public:

	std::list<unsigned int> **hashGrid;
	unsigned int hashGridSize;
	float invCellSize;
	unsigned int hashGridEntryCount;

	BBox hitPointsbbox;

	/**
	 * rebuild is done in log2 intervals of the iteration number.
	 * In single device, hash is rebuild at it 0,1,4,8,16,...
	 * In multi-device, each device will ensure that hash is in the
	 * correct log interval. e.g.: if it = 17 (log = 4) and last build was in
	 * it =6 (log = 2) , 4 > 2 than rebuild. If log <= lastBuild do not build.
	 */
	bool REBUILD_HASH;
	int lastBuildInterval;

	HashGridLookup(uint size, bool reb = true) :
			lookupAcc() {

		printf("Instancing a HashGridLookup of %d buckets\n", size);


		hashGridSize = size;
		hashGrid = NULL;
		//workerHitPointsInfo = NULL;

		lastBuildInterval = -1;

		REBUILD_HASH = reb;

		call_times=0;

	}

//	void setHitpoints(HitPointPositionInfo* d,HitPointRadianceFlux* workerHitPoints_) {
//		workerHitPointsInfo = d;
//		workerHitPoints = workerHitPoints_;
//	}

	inline void setBBox(BBox d) {
		hitPointsbbox = d;
	}

	inline BBox* getBBox() {
		return &(hitPointsbbox);
	}

	~HashGridLookup() {

	}

	void Build(float currentPhotonRadius2,

	HitPoint *workerHitPointsInfo);

	void AddFlux(const Point &hitPoint, const Normal &shadeN, const Vector wi,
			const Spectrum photonFlux, float currentPhotonRadius2,
			HitPoint *workerHitPointsInfo, PointerFreeScene *ss);

	inline void SplatFlux(const float dist2, HitPoint *hp,
			const float currentPhotonRadius2, const Normal &shadeN,
			const Vector wi, const Spectrum photonFlux, PointerFreeScene *ss);

	void UpdateQueryRange(float currentPhotonRadius2, uint it,
			HitPoint *workerHitPointsInfo);

private:
	unsigned int Hash(const int ix, const int iy, const int iz) {
		return (unsigned int) ((ix * 73856093) ^ (iy * 19349663)
				^ (iz * 83492791)) % hashGridSize;
	}

};

/**
 * sort by hash, hash function hash grid with neighbour pre-processing
 * cell size based on bbox and image resolution
 */
class GPUHashGrid: public lookupAcc {
public:

	/**
	 * For each hash bucket points in the PointIdx where the hitpoits
	 *  start that belong to this bucket
	 */
	int* FirstIdxBuff;
	/**
	 * Specifies for each bucket the number of hitpoints
	 */
	uint* NumHitpointsBuff;
	/**
	 * Sorted by hash bucket, specifies the index of the hitpoint
	 */
	uint* PointIdx;

	/**
	 * Power of two
	 */
	uint SpatialHashTableSize;

	BBox hitPointsbbox;

	/**
	 * rebuild is done in log2 intervals of the iteration number.
	 * In single device, hash is rebuild at it 0,1,4,8,16,...
	 * In multi-device, each device will ensure that hash is in the
	 * correct log interval. e.g.: if it = 17 (log = 4) and last build was in
	 * it =6 (log = 2) , 4 > 2 than rebuild. If log <= lastBuild do not build.
	 */
	bool REBUILD_HASH;
	int lastBuildInterval;

	GPUHashGrid(uint size, bool reb = true) {


		if (!IsPowerOfTwo(size)) {
			size = lower_power_of_two(size);
		}

		printf("Instancing a GPUHashGrid of %d buckets\n", size);


		SpatialHashTableSize = size;

		FirstIdxBuff = NULL;
		NumHitpointsBuff = NULL;
		PointIdx = NULL;

		REBUILD_HASH = reb;

		lastBuildInterval = -1;



	}

	void Init() {

		cudaMalloc(&FirstIdxBuff, sizeof(int) * SpatialHashTableSize);
		cudaMalloc(&NumHitpointsBuff, sizeof(uint) * SpatialHashTableSize);
		cudaMalloc(&PointIdx, sizeof(uint) * cfg->hitPointTotal * 8);
	}

	~GPUHashGrid();

	inline void setBBox(BBox d) {
		hitPointsbbox = d;
	}

	inline BBox* getBBox() {
		return &(hitPointsbbox);
	}

	void AddFlux(const Point &hitPoint, const Normal &shadeN, const Vector wi,
			const Spectrum photonFlux, float currentPhotonRadius2,
			HitPoint *workerHitPointsInfo, PointerFreeScene *ss) {

	}

	void UpdateQueryRange(float currentPhotonRadius2, uint it,

	HitPoint *workerHitPointsInfo);

	void Build(float currentPhotonRadius2, HitPoint *workerHitPointsInfo);

	void LookupPhotonHits(unsigned long long photonHitCount,
			float currentPhotonRadius2);

};

/**
 * sort by hash, morton code as hash function hash grid with
 * neighbour pre-processing
 * cell size based on bbox and image resolution
 */
class GPUMortonHashGrid: public lookupAcc {
public:

	/**
	 * For each hash bucket points in the PointIdx where the hitpoits
	 *  start that belong to this bucket
	 */
	int* FirstIdxBuff;
	/**
	 * Specifies for each bucket the number of hitpoints
	 */
	uint* NumHitpointsBuff;
	/**
	 * Sorted by hash bucket, specifies the index of the hitpoint
	 */
	//uint* PointIdx;

	/**
	 * Power of two
	 */
	uint SpatialHashTableSize;

	BBox hitPointsbbox;

	/**
	 * rebuild is done in log2 intervals of the iteration number.
	 * In single device, hash is rebuild at it 0,1,4,8,16,...
	 * In multi-device, each device will ensure that hash is in the
	 * correct log interval. e.g.: if it = 17 (log = 4) and last build was in
	 * it =6 (log = 2) , 4 > 2 than rebuild. If log <= lastBuild do not build.
	 */
	bool REBUILD_HASH;
	int lastBuildInterval;

	GPUMortonHashGrid(uint size, bool reb = true) {




		if (!IsPowerOfTwo(size)) {
			size = lower_power_of_two(size);
		}

		printf("Instancing a GPUMortonHashGrid of %d buckets\n", size);

		SpatialHashTableSize = size;

		FirstIdxBuff = NULL;
		NumHitpointsBuff = NULL;


		REBUILD_HASH = reb;

		lastBuildInterval = -1;




	}

	void Init() {

		cudaMalloc(&FirstIdxBuff, sizeof(int) * SpatialHashTableSize);
		cudaMalloc(&NumHitpointsBuff, sizeof(uint) * SpatialHashTableSize);
		//cudaMalloc(&PointIdx, sizeof(uint) * cfg->hitPointTotal * 8);
	}

	~GPUMortonHashGrid();

	inline void setBBox(BBox d) {
		hitPointsbbox = d;
	}

	inline BBox* getBBox() {
		return &(hitPointsbbox);
	}

	void AddFlux(const Point &hitPoint, const Normal &shadeN, const Vector wi,
			const Spectrum photonFlux, float currentPhotonRadius2,
			HitPoint *workerHitPointsInfo, PointerFreeScene *ss) {

	}

	void UpdateQueryRange(float currentPhotonRadius2, uint it,

	HitPoint *workerHitPointsInfo);

	void Build(float currentPhotonRadius2, HitPoint *workerHitPointsInfo);

	void LookupPhotonHits(unsigned long long photonHitCount,
			float currentPhotonRadius2);

};

/**
 * sort by morton, morton grid without
 * neighbour pre-processing
 * cell size based morton bits
 */
class GPUMortonGrid: public lookupAcc {
public:

	/**
	 * For each hash bucket points in the PointIdx where the hitpoits
	 *  start that belong to this bucket
	 */
	int* FirstIdxBuff;
	/**
	 * Specifies for each bucket the number of hitpoints
	 */
	uint* NumHitpointsBuff;

	/**
	 * Power of two
	 */
	uint MortonBlockCount;

	BBox hitPointsbbox;

	int lastBuildInterval;

	uint bits_per_dim;

	GPUMortonGrid(uint bits, bool reb = false) :
			bits_per_dim(bits) {

		printf("Instancing a GPUMortonGrid\n");


		MortonBlockCount = 1 << (3 * bits);

		printf("Instancing a GPUMortonGrid of %d buckets\n", MortonBlockCount);


		FirstIdxBuff = NULL;
		NumHitpointsBuff = NULL;

		lastBuildInterval = -1;

	}

	void Init() {

		cudaMalloc(&FirstIdxBuff, sizeof(int) * MortonBlockCount);
		cudaMalloc(&NumHitpointsBuff, sizeof(uint) * MortonBlockCount);

	}

	~GPUMortonGrid();

	inline void setBBox(BBox d) {
		hitPointsbbox = d;
	}

	inline BBox* getBBox() {
		return &(hitPointsbbox);
	}

	void AddFlux(const Point &hitPoint, const Normal &shadeN, const Vector wi,
			const Spectrum photonFlux, float currentPhotonRadius2,
			HitPoint *workerHitPointsInfo, PointerFreeScene *ss) {

	}

	void UpdateQueryRange(float currentPhotonRadius2, uint it,
			HitPoint *workerHitPointsInfo);

	void Build(float currentPhotonRadius2, HitPoint *workerHitPointsInfo);

	void LookupPhotonHits(unsigned long long photonHitCount,
			float currentPhotonRadius2);

};

//------------------------------------------------------------------------------
// KdTree accelerator
//------------------------------------------------------------------------------

class KdTree: public lookupAcc {
public:

	KdTree();

	~KdTree();

//	void Update(float currentPhotonRadius2,
//			HitPointRadianceFlux *workerHitPoints,
//			HitPointPositionInfo *workerHitPointsInfo);

	void AddFlux(const Point &hitPoint, const Normal &shadeN, const Vector wi,
			const Spectrum photonFlux, float currentPhotonRadius2,
			HitPoint *workerHitPointsInfo, PointerFreeScene *ss);

	inline void setBBox(BBox d) {
		hitPointsbbox = d;
	}

	inline BBox* getBBox() {
		return &(hitPointsbbox);
	}

	struct KdNode {
		void init(const float p, const unsigned int a) {
			splitPos = p;
			splitAxis = a;
			// Dade - in order to avoid a gcc warning
			rightChild = 0;
			rightChild = ~rightChild;
			hasLeftChild = 0;
		}

		void initLeaf() {
			splitAxis = 3;
			// Dade - in order to avoid a gcc warning
			rightChild = 0;
			rightChild = ~rightChild;
			hasLeftChild = 0;
		}

		// KdNode Data
		float splitPos;
		unsigned int splitAxis :2;
		unsigned int hasLeftChild :1;
		unsigned int rightChild :29;
	};

	struct CompareNode {
		CompareNode(int a) {
			axis = a;
		}

		int axis;

		bool operator()(const HitPoint *d1, const HitPoint *d2) const;
	};

	void RecursiveBuild(const unsigned int nodeNum, const unsigned int start,
			const unsigned int end, std::vector<HitPoint *> &buildNodes);

	inline void UpdateQueryRange(float currentPhotonRadius2, uint it,
			HitPoint *workerHitPointsInfo);

	void Build(float currentPhotonRadius2, HitPoint *workerHitPointsInfo);

	inline void SplatFlux(const float dist2, HitPoint *hp,
			const float currentPhotonRadius2, const Normal &shadeN,
			const Vector wi, const Spectrum photonFlux, PointerFreeScene *ss);

	HitPoint *hitPoints;

	KdNode *nodes;
	HitPoint **nodeData;
	unsigned int nNodes, nextFreeNode;
	float maxDistSquared;

	BBox hitPointsbbox;

};

/**
 * deprecated
 */
//class PointerFreeHashGrid: public lookupAcc {
//public:
//
//	std::list<unsigned int> **hashGrid;
//
//	unsigned int* hashGridLists;
//	unsigned int* hashGridLenghts;
//	unsigned int* hashGridListsIndex;
//
//	unsigned int hashGridSize;
//	float invCellSize;
//	unsigned int hashGridEntryCount;
//
//	unsigned int* hashGridListsBuff;
//	unsigned int* hashGridLenghtsBuff;
//	unsigned int* hashGridListsIndexBuff;
//
//	BBox hitPointsbbox;
//
//	bool firstTime;
//
//	PointerFreeHashGrid(uint size) {
//
//		hashGridSize = size;
//
//		hashGrid = NULL;
//
//		hashGridLenghts = NULL;
//		hashGridLists = NULL;
//		hashGridListsIndex = NULL;
//
//		hashGridLenghtsBuff = NULL;
//		hashGridListsBuff = NULL;
//		hashGridListsIndexBuff = NULL;
//
//		firstTime = true;
//	}
//
//	~PointerFreeHashGrid() {
//
//	}
//
//	inline void setBBox(BBox d) {
//		hitPointsbbox = d;
//	}
//
//	inline BBox* getBBox() {
//		return &(hitPointsbbox);
//	}
//
//	void Build(float currentPhotonRadius2,
//			HitPointRadianceFlux *workerHitPoints,
//			HitPointPositionInfo *workerHitPointsInfo);
//
//	void AddFlux(HitPointRadianceFlux *workerHitPoints, const Point &hitPoint,
//			const Normal &shadeN, const Vector wi, const Spectrum photonFlux,
//			float currentPhotonRadius2,
//			HitPointPositionInfo *workerHitPointsInfo, PointerFreeScene *ss) {
//
//	}
//
//	void UpdateQueryRange(float currentPhotonRadius2,
//			HitPointRadianceFlux *workerHitPoints,
//			HitPointPositionInfo *workerHitPointsInfo);
//
//	void updateLookupTable();
//
//private:
//	unsigned int Hash(const int ix, const int iy, const int iz) {
//		return (unsigned int) ((ix * 73856093) ^ (iy * 19349663)
//				^ (iz * 83492791)) % hashGridSize;
//	}
//
//};
#endif /* LOOKUPACC_H_ */
