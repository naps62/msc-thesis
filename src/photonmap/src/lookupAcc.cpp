/*
 * lookupAcc.cpp
 *
 *  Created on: Nov 9, 2012
 *      Author: rr
 */

#include "lookupAcc.h"
#include "cppbench.h"

void BuildMortonGrid_wrapper(GPUMortonGrid* lookup, HitPoint* hitpointsBuff,
		unsigned int hitPointsCount, float* BBpMin, float currentPhotonRadius2);

void BuildHashGrid_wrapper(GPUHashGrid* lookup, HitPoint* hitpointsBuff,
		unsigned int hitPointsCount, float* BBpMin, float currentPhotonRadius2);

void LookupMortonGridKernel_wrapper(GPUMortonGrid* lookupA, uint phitcount,
		engineType eT, float currentPhotonRadius2);

void LookupHashGridKernel_wrapper(GPUHashGrid* lookupA, uint phitcount,
		engineType eT, float currentPhotonRadius2) ;

void BuildMortonHashGrid_wrapper(GPUMortonHashGrid* lookup, HitPoint* hitpointsBuff,
		unsigned int hitPointsCount, float* BBpMin, float currentPhotonRadius2);

void LookupMortonHashGridKernel_wrapper(GPUMortonHashGrid* lookupA, uint phitcount,
		engineType eT, float currentPhotonRadius2);

lookupAcc::lookupAcc() {
	// TODO Auto-generated constructor stub
	call_times = 0;
}

lookupAcc::~lookupAcc() {
	// TODO Auto-generated destructor stub
}

//------------------------------------------------------------------------------
// HashGridLookup accelerator
//------------------------------------------------------------------------------

void HashGridLookup::UpdateQueryRange(float currentPhotonRadius2, uint it,
		HitPoint *workerHitPointsInfo) {

	double itd = (double) it;

	int logit = (int) log2(itd);

	if (logit == 0 || (REBUILD_HASH && logit > lastBuildInterval)
			|| lastBuildInterval == -1) {
		Build(currentPhotonRadius2, workerHitPointsInfo);
		lastBuildInterval = logit;
	}

}

void HashGridLookup::Build(float currentPhotonRadius2,
		HitPoint *workerHitPointsInfo) {

	hitPointsbbox.Expand(sqrt(currentPhotonRadius2));

	const unsigned int hitPointsCount = cfg->hitPointTotal;
	const BBox &hpBBox = hitPointsbbox;

	float maxPhotonRadius2 = currentPhotonRadius2;

	const float cellSize = sqrtf(maxPhotonRadius2) * 2.f;
	invCellSize = 1.f / cellSize;

	if (!hashGrid) {
		hashGrid = new std::list<uint>*[hashGridSize];

		for (unsigned int i = 0; i < hashGridSize; ++i)
			hashGrid[i] = NULL;
	} else {
		for (unsigned int i = 0; i < hashGridSize; ++i) {
			delete hashGrid[i];
			hashGrid[i] = NULL;
		}
	}

//	double lastPrintTime = WallClockTime();
	unsigned long long entryCount = 0;

	for (unsigned int i = 0; i < hitPointsCount; ++i) {

//		if (WallClockTime() - lastPrintTime > 2.0) {
//			std::cerr << "  " << i / 1000 << "k/" << hitPointsCount / 1000
//					<< "k" << std::endl;
//			lastPrintTime = WallClockTime();
//		}

		HitPoint *hp = &workerHitPointsInfo[i];

		if (hp->type == SURFACE) {

			float photonRadius;
			if (cfg->GetEngineType() == PPMPA || cfg->GetEngineType() == SPPMPA)
				photonRadius = sqrtf(currentPhotonRadius2);

			else {
				HitPoint *hpp = &workerHitPointsInfo[i];
				photonRadius = sqrtf(hpp->accumPhotonRadius2);
			}

			const Vector rad(photonRadius, photonRadius, photonRadius);
			const Vector bMin = ((hp->position - rad) - hpBBox.pMin)
					* invCellSize;
			const Vector bMax = ((hp->position + rad) - hpBBox.pMin)
					* invCellSize;

			for (int iz = abs(int(bMin.z)); iz <= abs(int(bMax.z)); iz++) {
				for (int iy = abs(int(bMin.y)); iy <= abs(int(bMax.y)); iy++) {
					for (int ix = abs(int(bMin.x)); ix <= abs(int(bMax.x));
							ix++) {

						int hv = Hash(ix, iy, iz);

						if (hashGrid[hv] == NULL)
							hashGrid[hv] = new std::list<uint>();

						hashGrid[hv]->push_front(i);
						++entryCount;
					}
				}
			}
		}
	}

	hashGridEntryCount = entryCount;

	//std::cerr << "Max. hit points in a single hash grid entry: " << maxPathCount << std::endl;
	//std::cerr << "Total hash grid entry: " << entryCount << std::endl;
	//std::cerr << "Avg. hit points in a single hash grid entry: "
	//		<< entryCount / hashGridSize << std::endl;

	//printf("Sizeof %d\n", sizeof(HitPoint*));

	// HashGrid debug code
	/*for (unsigned int i = 0; i < hashGridSize; ++i) {
	 if (hashGrid[i]) {
	 if (hashGrid[i]->size() > 10) {
	 std::cerr << "HashGrid[" << i << "].size() = " <<hashGrid[i]->size() << std::endl;
	 }
	 }
	 }*/
}

inline void HashGridLookup::SplatFlux(const float dist2, HitPoint *hp,
		const float currentPhotonRadius2, const Normal &shadeN, const Vector wi,
		const Spectrum photonFlux, PointerFreeScene *ss) {


	if (cfg->GetEngineType() == PPM || cfg->GetEngineType() == SPPM) {
		if (dist2 > hp->accumPhotonRadius2)
			return;
	} else {

		if (dist2 > currentPhotonRadius2)
			return;
	}

	const float dot = Dot(hp->normal, wi);
	if (dot <= 0.0001f)
		return;

	__sync_fetch_and_add(&hp->accumPhotonCount, 1);
	__sync_fetch_and_add(&call_times, 1);



	Spectrum f;

	POINTERFREESCENE::Material *hitPointMat = &ss->materials[hp->materialSS];

	switch (hitPointMat->type) {

	case MAT_MATTE:
		ss->Matte_f(&hitPointMat->param.matte, hp->wo, wi, shadeN, f);
		break;

	case MAT_MATTEMIRROR:
		ss->MatteMirror_f(&hitPointMat->param.matteMirror, hp->wo, wi, shadeN,
				f);
		break;

	case MAT_MATTEMETAL:
		ss->MatteMetal_f(&hitPointMat->param.matteMetal, hp->wo, wi, shadeN, f);
		break;

	case MAT_ALLOY:
		ss->Alloy_f(&hitPointMat->param.alloy, hp->wo, wi, shadeN, f);
		break;
	default:
		break;

	}

	//different in smallux SPPM
	Spectrum flux = photonFlux * hp->throughput * f
	//darkening?
			* AbsDot(shadeN, wi);

#pragma omp critical
	{
		hp->accumReflectedFlux = (hp->accumReflectedFlux + flux);
	}

//#pragma omp atomic
//		ihp->accumReflectedFlux.r+= flux.r;
//#pragma omp atomic
//		ihp->accumReflectedFlux.g += flux.g;
//#pragma omp atomic
//		ihp->accumReflectedFlux.b += flux.b;

}

void HashGridLookup::AddFlux(const Point &hitPoint, const Normal &shadeN,
		const Vector wi, const Spectrum photonFlux, float currentPhotonRadius2,
		HitPoint *workerHitPointsInfo, PointerFreeScene *ss) {

	// Look for eye path hit points near the current hit point
	Vector hh = (hitPoint - hitPointsbbox.pMin) * invCellSize;
	const int ix = abs(int(hh.x));
	const int iy = abs(int(hh.y));
	const int iz = abs(int(hh.z));

	//	std::list<uint> *hps = hashGrid[Hash(ix, iy, iz, hashGridSize)];
	//	if (hps) {
	//		std::list<uint>::iterator iter = hps->begin();
	//		while (iter != hps->end()) {
	//
	//			HitPoint *hp = &hitPoints[*iter++];

	uint gridEntry = Hash(ix, iy, iz);
	std::list<unsigned int>* hps = hashGrid[gridEntry];

	if (hps) {
		std::list<unsigned int>::iterator iter = hps->begin();
		while (iter != hps->end()) {

			HitPoint *hp = &workerHitPointsInfo[*iter++];

			//HitPointRadianceFlux *ihp = &workerHitPoints[*iter++];

			Vector v = hp->position - hitPoint;

			const float dist2 = DistanceSquared(hp->position, hitPoint);

			SplatFlux(dist2, hp, currentPhotonRadius2, shadeN, wi, photonFlux,
					ss);

		}

	}
}

//------------------------------------------------------------------------------
// HashGrid accelerator in the GPU
//------------------------------------------------------------------------------

GPUHashGrid::~GPUHashGrid() {

}

void GPUHashGrid::Build(float currentPhotonRadius2,
		HitPoint *workerHitPointsInfo) {

	hitPointsbbox.Expand(sqrt(currentPhotonRadius2));

	BuildHashGrid_wrapper(this, workerHitPointsInfo, cfg->hitPointTotal,
			(float*) &hitPointsbbox.pMin, currentPhotonRadius2);
}

void GPUHashGrid::UpdateQueryRange(float currentPhotonRadius2, uint it,

HitPoint *workerHitPointsInfo) {

	double itd = (double) it;

	int logit = (int) log2(itd);

	if (logit == 0 || (REBUILD_HASH && logit > lastBuildInterval)
			|| lastBuildInterval == -1) {
		Build(currentPhotonRadius2, workerHitPointsInfo);
		lastBuildInterval = logit;
	}
}

void GPUHashGrid::LookupPhotonHits(unsigned long long photonHitCount,
		float currentPhotonRadius2) {
	LookupHashGridKernel_wrapper(this, photonHitCount, cfg->enginetype,
			currentPhotonRadius2);
}




//------------------------------------------------------------------------------
// MortonHashGrid accelerator in the GPU
//------------------------------------------------------------------------------

GPUMortonHashGrid::~GPUMortonHashGrid() {

}

void GPUMortonHashGrid::Build(float currentPhotonRadius2,
		HitPoint *workerHitPointsInfo) {

	hitPointsbbox.Expand(sqrt(currentPhotonRadius2));

	BuildMortonHashGrid_wrapper(this, workerHitPointsInfo, cfg->hitPointTotal,
			(float*) &hitPointsbbox.pMin, currentPhotonRadius2);
}

void GPUMortonHashGrid::UpdateQueryRange(float currentPhotonRadius2, uint it,

HitPoint *workerHitPointsInfo) {

	double itd = (double) it;

	int logit = (int) log2(itd);

	if (logit == 0 || (REBUILD_HASH && logit > lastBuildInterval)
			|| lastBuildInterval == -1) {
		Build(currentPhotonRadius2, workerHitPointsInfo);
		lastBuildInterval = logit;
	}
}

void GPUMortonHashGrid::LookupPhotonHits(unsigned long long photonHitCount,
		float currentPhotonRadius2) {
	LookupMortonHashGridKernel_wrapper(this, photonHitCount, cfg->enginetype,
			currentPhotonRadius2);
}


//------------------------------------------------------------------------------
// MortonGrid accelerator in the GPU
//------------------------------------------------------------------------------

GPUMortonGrid::~GPUMortonGrid() {

}

void GPUMortonGrid::Build(float currentPhotonRadius2,
		HitPoint *workerHitPointsInfo) {

	hitPointsbbox.Expand(sqrt(currentPhotonRadius2));

	BuildMortonGrid_wrapper(this, workerHitPointsInfo, cfg->hitPointTotal,
			(float*) &hitPointsbbox.pMin, currentPhotonRadius2);

}

void GPUMortonGrid::UpdateQueryRange(float currentPhotonRadius2, uint it,
		HitPoint *workerHitPointsInfo) {

	if (lastBuildInterval == -1) {
		Build(currentPhotonRadius2, workerHitPointsInfo);
		lastBuildInterval = it;
	}
}

void GPUMortonGrid::LookupPhotonHits(unsigned long long photonHitCount,
		float currentPhotonRadius2) {
	LookupMortonGridKernel_wrapper(this, photonHitCount, cfg->enginetype,
			currentPhotonRadius2);
}

//------------------------------------------------------------------------------
// KdTree accelerator
//------------------------------------------------------------------------------

KdTree::KdTree() {
	//hitPoints = engine->GetHitPointInfo(0);
	nNodes = cfg->hitPointTotal;
	nextFreeNode = 1;
	nodes = NULL;
	nodeData = NULL;

}

KdTree::~KdTree() {
	delete[] nodes;
	delete[] nodeData;
}

bool KdTree::CompareNode::operator ()(const HitPoint *d1,
		const HitPoint *d2) const {
	return (d1->position[axis] == d2->position[axis]) ?
			(d1 < d2) : (d1->position[axis] < d2->position[axis]);
}

/**
 * Split position corresponds to a hitpoint.
 * left chidl node is always next to current node in the array, due to rescursivity.
 * Next split axis is not necessarly the orthogonal one (?)
 */
void KdTree::RecursiveBuild(const unsigned int nodeNum,
		const unsigned int start, const unsigned int end,
		std::vector<HitPoint *> &buildNodes) {
	assert(nodeNum >= 0);
	assert(start >= 0);
	assert(end >= 0);
	assert(nodeNum < nNodes);
	assert(start < nNodes);
	assert(end <= nNodes);

	// Create leaf node of kd-tree if we've reached the bottom
	if (start + 1 == end) {
		nodes[nodeNum].initLeaf();
		nodeData[nodeNum] = buildNodes[start];

		return;
	}

	// Choose split direction and partition data
	// Compute bounds of data from start to end
	BBox bound;
	for (unsigned int i = start; i < end; ++i)
		bound = Union(bound, buildNodes[i]->position);
	unsigned int splitAxis = bound.MaximumExtent();
	unsigned int splitPos = (start + end) / 2;

	std::nth_element(buildNodes.begin() + start, buildNodes.begin() + splitPos,
			buildNodes.begin() + end, CompareNode(splitAxis));

	// Allocate kd-tree node and continue recursively
	nodes[nodeNum].init(buildNodes[splitPos]->position[splitAxis], splitAxis);
	nodeData[nodeNum] = buildNodes[splitPos];

	if (start < splitPos) {
		nodes[nodeNum].hasLeftChild = 1;
		const unsigned int childNum = nextFreeNode++;
		RecursiveBuild(childNum, start, splitPos, buildNodes);
	}

	if (splitPos + 1 < end) {
		nodes[nodeNum].rightChild = nextFreeNode++;
		RecursiveBuild(nodes[nodeNum].rightChild, splitPos + 1, end,
				buildNodes);
	}
}

void KdTree::UpdateQueryRange(float currentPhotonRadius2, uint it,
		HitPoint *workerHitPointsInfo) {

	if (nodes == NULL) {
		Build(currentPhotonRadius2, workerHitPointsInfo);
	}

	else {

		maxDistSquared = 0.f;
		if (cfg->GetEngineType() == PPM || cfg->GetEngineType() == SPPM) {
			for (unsigned int i = 0; i < nNodes; ++i) {
				maxDistSquared = Max(maxDistSquared,
						workerHitPointsInfo[i].accumPhotonRadius2);
			}

		}

		else
			maxDistSquared = currentPhotonRadius2;

		std::cerr << "kD-Tree search radius: " << sqrtf(maxDistSquared)
				<< std::endl;

	}

}

//void KdTree::Update(float currentPhotonRadius2,
//		HitPointRadianceFlux *workerHitPoints,
//		HitPointPositionInfo *workerHitPointsInfo) {
//
//	if (nodes == NULL) {
//		Build(currentPhotonRadius2, workerHitPoints, workerHitPointsInfo);
//	}
//
//	else {
//
//		if (cfg->GetEngineType() == PPM || cfg->GetEngineType() == SPPM) {
//			for (unsigned int i = 0; i < nNodes; ++i) {
//				workerHitPointsInfo[i].id = i;
//				buildNodes.push_back(&workerHitPointsInfo[i]);
//
//				if (cfg->GetEngineType() == PPM || cfg->GetEngineType() == SPPM) {
//					maxDistSquared = Max(maxDistSquared,
//							workerHitPoints[i].accumPhotonRadius2);
//				}
//
//			}
//		} else
//			maxDistSquared = currentPhotonRadius2;
//
//		std::cerr << "kD-Tree search radius: " << sqrtf(maxDistSquared)
//				<< std::endl;
//
//	}
//
//}

void KdTree::Build(float currentPhotonRadius2, HitPoint *workerHitPointsInfo) {
	delete[] nodes;
	delete[] nodeData;

	std::cerr << "Building kD-Tree with " << nNodes << " nodes" << std::endl;

	nodes = new KdNode[nNodes];
	nodeData = new HitPoint*[nNodes];
	nextFreeNode = 1;

// Begin the KdTree building process
	std::vector<HitPoint *> buildNodes;
	buildNodes.reserve(nNodes);
	maxDistSquared = 0.f;
	for (unsigned int i = 0; i < nNodes; ++i) {
		workerHitPointsInfo[i].id = i;
		buildNodes.push_back(&workerHitPointsInfo[i]);

		if (cfg->GetEngineType() == PPM || cfg->GetEngineType() == SPPM) {
			maxDistSquared = Max(maxDistSquared,
					workerHitPointsInfo[i].accumPhotonRadius2);
		}

	}

	if (cfg->GetEngineType() == PPMPA || cfg->GetEngineType() == SPPMPA)
		maxDistSquared = currentPhotonRadius2;

	std::cerr << "kD-Tree search radius: " << sqrtf(maxDistSquared)
			<< std::endl;

	RecursiveBuild(0, 0, nNodes, buildNodes);
	assert(nNodes == nextFreeNode);
}

/**
 * I think there are distances recalcualted and re-evalueated
 */
void KdTree::AddFlux(const Point &p, const Normal &shadeN, const Vector wi,
		const Spectrum photonFlux, float currentPhotonRadius2,
		HitPoint *workerHitPointsInfo, PointerFreeScene *ss) {

	unsigned int nodeNumStack[64];
// Start from the first node
	nodeNumStack[0] = 0;
	int stackIndex = 0;

	while (stackIndex >= 0) {
		const unsigned int nodeNum = nodeNumStack[stackIndex--];
		KdNode *node = &nodes[nodeNum];

		const int axis = node->splitAxis;
		if (axis != 3) {
			const float dist = p[axis] - node->splitPos;
			const float dist2 = dist * dist;
			if (p[axis] <= node->splitPos) { //RR:left

				if ((dist2 < maxDistSquared) && (node->rightChild < nNodes)) // if in range, there is a part of the search radius that is on the right side. If it exist, stack it. If not no need to process right side
					nodeNumStack[++stackIndex] = node->rightChild;

				if (node->hasLeftChild) // RR: stack cell to process
					nodeNumStack[++stackIndex] = nodeNum + 1;

			} else { //RR: Right

				if (node->rightChild < nNodes) // RR: if right node is cell
					nodeNumStack[++stackIndex] = node->rightChild; // RR: stack cell to process

				if ((dist2 < maxDistSquared) && (node->hasLeftChild)) // if in range, there is a part of the search radius that is on the left side. If it exist, stack it. If not no need to process left side
					nodeNumStack[++stackIndex] = nodeNum + 1;
			}
		}

		/*
		 * some distances were already evaluated ?, node->splitPos = nodeData[nodeNum]?
		 */

		// Process the leaf
		HitPoint *hp = nodeData[nodeNum];

		//HitPointRadianceFlux *ihp = &workerHitPointsInfo[hp->id];

		const float dist2 = DistanceSquared(hp->position, p);

		SplatFlux(dist2, hp, currentPhotonRadius2, shadeN, wi, photonFlux, ss);
	}
}

inline void KdTree::SplatFlux(const float dist2, HitPoint *hp,
		const float currentPhotonRadius2, const Normal &shadeN, const Vector wi,
		const Spectrum photonFlux, PointerFreeScene *ss) {

//#pragma omp atomic
//	call_times++;

	if (cfg->GetEngineType() == PPM || cfg->GetEngineType() == SPPM) {
		if (dist2 > hp->accumPhotonRadius2)
			return;
	} else {

		if (dist2 > currentPhotonRadius2)
			return;
	}

	const float dot = Dot(hp->normal, wi);
	if (dot <= 0.0001f)
		return;

	__sync_fetch_and_add(&hp->accumPhotonCount, 1);

	Spectrum f;

	POINTERFREESCENE::Material *hitPointMat = &ss->materials[hp->materialSS];

	switch (hitPointMat->type) {

	case MAT_MATTE:
		ss->Matte_f(&hitPointMat->param.matte, hp->wo, wi, shadeN, f);
		break;

	case MAT_MATTEMIRROR:
		ss->MatteMirror_f(&hitPointMat->param.matteMirror, hp->wo, wi, shadeN,
				f);
		break;

	case MAT_MATTEMETAL:
		ss->MatteMetal_f(&hitPointMat->param.matteMetal, hp->wo, wi, shadeN, f);
		break;

	case MAT_ALLOY:
		ss->Alloy_f(&hitPointMat->param.alloy, hp->wo, wi, shadeN, f);
		break;
	default:
		break;

	}

	//different in smallux SPPM
	Spectrum flux = photonFlux * hp->throughput * f
	//darkening?
			* AbsDot(shadeN, wi);

#pragma omp critical
	{
		hp->accumReflectedFlux = (hp->accumReflectedFlux + flux);
	}

//#pragma omp atomic
//		ihp->accumReflectedFlux.r+= flux.r;
//#pragma omp atomic
//		ihp->accumReflectedFlux.g += flux.g;
//#pragma omp atomic
//		ihp->accumReflectedFlux.b += flux.b;

}

//------------------------------------------------------------------------------
// PointerFreeHashGrid accelerator
//------------------------------------------------------------------------------

//void PointerFreeHashGrid::Build(float currentPhotonRadius2,
//		HitPointRadianceFlux *workerHitPoints,
//		HitPointPositionInfo *workerHitPointsInfo) {
//
//	const unsigned int hitPointsCount = cfg->hitPointTotal;
//	const BBox &hpBBox = hitPointsbbox;
//
//	// Calculate the size of the grid cell
//	float maxPhotonRadius2 = currentPhotonRadius2;
////	if (cfg->GetEngineType() == PPMPA || cfg->GetEngineType() == SPPMPA)
////
////		maxPhotonRadius2 = currentPhotonRadius2;
////	else {
////		maxPhotonRadius2 = 0.f;
////		for (unsigned int i = 0; i < hitPointsCount; ++i) {
////			HitPointPositionInfo *ihp = &workerHitPointsInfo[i];
////			HitPointRadianceFlux *hp = &workerHitPoints[i];
////
////			if (ihp->type == SURFACE)
////				maxPhotonRadius2 = Max(maxPhotonRadius2,
////						hp->accumPhotonRadius2);
////		}
////	}
//
//	const float cellSize = sqrtf(maxPhotonRadius2) * 2.f;
//	//std::cerr << "Hash grid cell size: " << cellSize << std::endl;
//	invCellSize = 1.f / cellSize;
//
//	// TODO: add a tunable parameter for hashgrid size
//	//hashGridSize = hitPointsCount;
//	if (!hashGrid) {
//		hashGrid = new std::list<uint>*[hashGridSize];
//
//		for (unsigned int i = 0; i < hashGridSize; ++i)
//			hashGrid[i] = NULL;
//	} else {
//		for (unsigned int i = 0; i < hashGridSize; ++i) {
//			delete hashGrid[i];
//			hashGrid[i] = NULL;
//		}
//	}
//
//	//std::cerr << "Building hit points hash grid:" << std::endl;
//	//std::cerr << "  0k/" << hitPointsCount / 1000 << "k" << std::endl;
//	//unsigned int maxPathCount = 0;
//	double lastPrintTime = WallClockTime();
//	unsigned long long entryCount = 0;
//
//	for (unsigned int i = 0; i < hitPointsCount; ++i) {
//
//		if (WallClockTime() - lastPrintTime > 2.0) {
//			std::cerr << "  " << i / 1000 << "k/" << hitPointsCount / 1000
//					<< "k" << std::endl;
//			lastPrintTime = WallClockTime();
//		}
//
//		HitPointPositionInfo *hp = &workerHitPointsInfo[i];
//
//		float photonRadius;
//		if (hp->type == SURFACE) {
//
//			if (cfg->GetEngineType() == PPMPA || cfg->GetEngineType() == SPPMPA)
//				photonRadius = sqrtf(currentPhotonRadius2);
//
//			else {
//				HitPointRadianceFlux *hpp = &workerHitPoints[i];
//				photonRadius = sqrtf(hpp->accumPhotonRadius2);
//
//			}
//			const Vector rad(photonRadius, photonRadius, photonRadius);
//			const Vector bMin = ((hp->position - rad) - hpBBox.pMin)
//					* invCellSize;
//			const Vector bMax = ((hp->position + rad) - hpBBox.pMin)
//					* invCellSize;
//
//			for (int iz = abs(int(bMin.z)); iz <= abs(int(bMax.z)); iz++) {
//				for (int iy = abs(int(bMin.y)); iy <= abs(int(bMax.y)); iy++) {
//					for (int ix = abs(int(bMin.x)); ix <= abs(int(bMax.x));
//							ix++) {
//
//						int hv = Hash(ix, iy, iz);
//
//						if (hashGrid[hv] == NULL)
//							hashGrid[hv] = new std::list<uint>();
//
//						hashGrid[hv]->push_front(i);
//						++entryCount;
//
//						/*// hashGrid[hv]->size() is very slow to execute
//						 if (hashGrid[hv]->size() > maxPathCount)
//						 maxPathCount = hashGrid[hv]->size();*/
//					}
//				}
//			}
//		}
//	}
//
//	hashGridEntryCount = entryCount;
//
//	//std::cerr << "Max. hit points in a single hash grid entry: " << maxPathCount << std::endl;
//	std::cerr << "Total hash grid entry: " << entryCount << std::endl;
//	std::cerr << "Avg. hit points in a single hash grid entry: "
//			<< entryCount / hashGridSize << std::endl;
//
//	//printf("Sizeof %d\n", sizeof(HitPoint*));
//
//	// HashGrid debug code
////	for (unsigned int i = 0; i < hashGridSize; ++i) {
////	 if (hashGrid[i]) {
////	 if (hashGrid[i]->size() > 10) {
////	 std::cerr << "HashGrid[" << i << "].size() = " <<hashGrid[i]->size() << std::endl;
////	 }
////	 }
////	 }
//}
//
//void PointerFreeHashGrid::updateLookupTable() {
//
//	if (hashGridLists)
//		delete[] hashGridLists;
//
//	hashGridLists = new uint[hashGridEntryCount];
//
//	if (hashGridLenghts)
//		memset(hashGridLenghts, 0, hashGridSize * sizeof(uint));
//	else
//		hashGridLenghts = new uint[hashGridSize];
//
//	if (hashGridListsIndex)
//		memset(hashGridListsIndex, 0, hashGridSize * sizeof(uint));
//	else
//		hashGridListsIndex = new uint[hashGridSize];
//
//	uint listIndex = 0;
//	for (unsigned int i = 0; i < hashGridSize; ++i) {
//
//		std::list<uint> *hps = hashGrid[i];
//
//		hashGridListsIndex[i] = listIndex;
//
//		if (hps) {
//			hashGridLenghts[i] = hps->size();
//			std::list<uint>::iterator iter = hps->begin();
//			while (iter != hps->end()) {
//				hashGridLists[listIndex++] = *iter++;
//
//			}
//		} else {
//			hashGridLenghts[i] = 0;
//		}
//
//	}
//
//	//checkCUDAmemory("before updateLookupTable");
//
//	uint size1 = sizeof(uint) * hashGridEntryCount;
//
//	if (hashGridListsBuff)
//		cudaFree(hashGridListsBuff);
//	cudaMalloc((void**) (&hashGridListsBuff), size1);
//
//	cudaMemset(hashGridListsBuff, 0, size1);
//	cudaMemcpy(hashGridListsBuff, hashGridLists, size1, cudaMemcpyHostToDevice);
//
//	uint size2 = sizeof(uint) * hashGridSize;
//
//	if (!hashGridListsIndexBuff)
//		cudaMalloc((void**) (&hashGridListsIndexBuff), size2);
//
//	cudaMemset(hashGridListsIndexBuff, 0, size2);
//
//	cudaMemcpy(hashGridListsIndexBuff, hashGridListsIndex, size2,
//			cudaMemcpyHostToDevice);
//
//	if (!hashGridLenghtsBuff)
//		cudaMalloc((void**) (&hashGridLenghtsBuff), size2);
//
//	cudaMemset(hashGridLenghtsBuff, 0, size2);
//
//	cudaMemcpy(hashGridLenghtsBuff, hashGridLenghts, size2,
//			cudaMemcpyHostToDevice);
//
//	checkCUDAError();
//
//	checkCUDAmemory((char*) "After updateLookupTable");
//
//}
//
//void PointerFreeHashGrid::UpdateQueryRange(float currentPhotonRadius2,
//		HitPointRadianceFlux *workerHitPoints,
//		HitPointPositionInfo *workerHitPointsInfo) {
//
//#ifndef REBUILD_HASH
//	if (firstTime
//			&& (cfg->GetEngineType() == PPM || cfg->GetEngineType() == PPMPA))
//#endif
//		Build(currentPhotonRadius2, workerHitPoints, workerHitPointsInfo);
//
//	firstTime = false;
//
//}
