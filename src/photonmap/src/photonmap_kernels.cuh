__device__ uint hits;

struct HashParams
{
	float cellSize;
	float3 bbMin;
	float3 invCellSize;
	uint SpatialHashTableSize;
	engineType eT;

};

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z,
		uint bits_per_axis) {
	float max_cells = 1 << bits_per_axis;
	x = min(max(x * max_cells, 0.0f), max_cells - 1);
	y = min(max(y * max_cells, 0.0f), max_cells - 1);
	z = min(max(z * max_cells, 0.0f), max_cells - 1);
	unsigned int xx = expandBits((unsigned int) x);
	unsigned int yy = expandBits((unsigned int) y);
	unsigned int zz = expandBits((unsigned int) z);
	return xx * 4 + yy * 2 + zz;
}

__constant__ HashParams g_Params;
/*
 * avoids passing constant pointers by kernel parameters
 */
__constant__ CUDA_Worker* workerBuff_c;
__constant__ PointerFreeScene* ssBuff_c;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}
};

/**
 * reduces to an array of nblocks length. op 0 = min, op 1 = max
 */
template<class T>
__global__ void reduce2(float *g_idata, T* min, unsigned int n, uint op) {
	T *sdata = SharedMemory<T>();

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i] : 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			//sdata[tid] += sdata[tid + s];
			if (op == 0)
				sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
			else
				sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		min[blockIdx.x] = sdata[0];
}

template<class T, uint p>
__global__ void reduceToFloat(T *g_idata, float* min, unsigned int n, uint op) {

	float *sdata = SharedMemory<float>();

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i].GetPosition(p) : 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			//sdata[tid] += sdata[tid + s];
			if (op == 0)
				sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
			else
				sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		min[blockIdx.x] = sdata[0];
}

/*
 * extact radius from struct hitpoint to continous array.
 */__global__ void BoilRadius2(HitPoint *hitPointInfo, uint count, float* out) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < count && hitPointInfo[i].type != CONSTANT_COLOR)
		out[i] = hitPointInfo[i].accumPhotonRadius2;

}

/*
 * extact position[0,1,2] from struct hitpoint to continous array.
 */
template<class T, uint p>
__global__ void BoilPosition(T *g_idata, uint count, float* out) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < count && g_idata[i].GetType() != CONSTANT_COLOR)
		out[i] = g_idata[i].GetPosition(p);

}

__global__ void SetNonPAInitialRadius2(HitPoint *g_idata, uint count,
		float rad2) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < count)
		g_idata[i].accumPhotonRadius2 = rad2;

}

// Using invDir0/invDir1/invDir2 and sign0/sign1/sign2 instead of an
// array because I dont' trust OpenCL compiler =)

__device__ int4 QBVHNode_BBoxIntersect(const float4 bboxes_minX,
		const float4 bboxes_maxX, const float4 bboxes_minY,
		const float4 bboxes_maxY, const float4 bboxes_minZ,
		const float4 bboxes_maxZ, const POINTERFREESCENE::QuadRay *ray4,
		const float4 invDir0, const float4 invDir1, const float4 invDir2,
		const int signs0, const int signs1, const int signs2) {
	float4 tMin = ray4->mint;
	float4 tMax = ray4->maxt;

	// X coordinate
	tMin = fmaxf(tMin, (bboxes_minX - ray4->ox) * invDir0);
	tMax = fminf(tMax, (bboxes_maxX - ray4->ox) * invDir0);

	// Y coordinate
	tMin = fmaxf(tMin, (bboxes_minY - ray4->oy) * invDir1);
	tMax = fminf(tMax, (bboxes_maxY - ray4->oy) * invDir1);

	// Z coordinate
	tMin = fmaxf(tMin, (bboxes_minZ - ray4->oz) * invDir2);
	tMax = fminf(tMax, (bboxes_maxZ - ray4->oz) * invDir2);

	// Return the visit flags
	return (tMax >= tMin);
}

__device__ void QuadTriangle_Intersect(const float4 origx, const float4 origy,
		const float4 origz, const float4 edge1x, const float4 edge1y,
		const float4 edge1z, const float4 edge2x, const float4 edge2y,
		const float4 edge2z, const uint4 primitives,
		POINTERFREESCENE::QuadRay *ray4, RayHit *rayHit) {
	//--------------------------------------------------------------------------
	// Calc. b1 coordinate

	const float4 s1x = (ray4->dy * edge2z) - (ray4->dz * edge2y);
	const float4 s1y = (ray4->dz * edge2x) - (ray4->dx * edge2z);
	const float4 s1z = (ray4->dx * edge2y) - (ray4->dy * edge2x);

	const float4 divisor = (s1x * edge1x) + (s1y * edge1y) + (s1z * edge1z);

	const float4 dx = ray4->ox - origx;
	const float4 dy = ray4->oy - origy;
	const float4 dz = ray4->oz - origz;

	const float4 b1 = ((dx * s1x) + (dy * s1y) + (dz * s1z)) / divisor;

	//--------------------------------------------------------------------------
	// Calc. b2 coordinate

	const float4 s2x = (dy * edge1z) - (dz * edge1y);
	const float4 s2y = (dz * edge1x) - (dx * edge1z);
	const float4 s2z = (dx * edge1y) - (dy * edge1x);

	const float4 b2 = ((ray4->dx * s2x) + (ray4->dy * s2y) + (ray4->dz * s2z))
			/ divisor;

	//--------------------------------------------------------------------------
	// Calc. b0 coordinate

	const float4 b0 = (make_float4(1.f)) - b1 - b2;

	//--------------------------------------------------------------------------

	const float4 t = ((edge2x * s2x) + (edge2y * s2y) + (edge2z * s2z))
			/ divisor;

	float _b1, _b2;
	float maxt = ray4->maxt.x;
	uint index;

	int4 cond = (divisor != make_float4(0.f)) & (b0 >= make_float4(0.f))
			& (b1 >= make_float4(0.f)) & (b2 >= make_float4(0.f))
			& (t > ray4->mint);

	const int cond0 = cond.x && (t.x < maxt);
	maxt = select(maxt, t.x, cond0);
	_b1 = select(0.f, b1.x, cond0);
	_b2 = select(0.f, b2.x, cond0);
	index = select(0xffffffffu, primitives.x, cond0);

	const int cond1 = cond.y && (t.y < maxt);
	maxt = select(maxt, t.y, cond1);
	_b1 = select(_b1, b1.y, cond1);
	_b2 = select(_b2, b2.y, cond1);
	index = select(index, primitives.y, cond1);

	const int cond2 = cond.z && (t.z < maxt);
	maxt = select(maxt, t.z, cond2);
	_b1 = select(_b1, b1.z, cond2);
	_b2 = select(_b2, b2.z, cond2);
	index = select(index, primitives.z, cond2);

	const int cond3 = cond.w && (t.w < maxt);
	maxt = select(maxt, t.w, cond3);
	_b1 = select(_b1, b1.w, cond3);
	_b2 = select(_b2, b2.w, cond3);
	index = select(index, primitives.w, cond3);

	if (index == 0xffffffffu)
		return;

	ray4->maxt = make_float4(maxt);

	rayHit->t = maxt;
	rayHit->b1 = _b1;
	rayHit->b2 = _b2;
	rayHit->index = index;
}

__device__ void subIntersect(Ray& ray, POINTERFREESCENE::QBVHNode *nodes,
		POINTERFREESCENE::QuadTriangle *quadTris, RayHit& rayHit) {

	// Prepare the ray for intersection
	POINTERFREESCENE::QuadRay ray4;
	{
		float4 *basePtr = (float4 *) &ray;
		float4 data0 = (*basePtr++);
		float4 data1 = (*basePtr);

		ray4.ox = make_float4(data0.x);
		ray4.oy = make_float4(data0.y);
		ray4.oz = make_float4(data0.z);

		ray4.dx = make_float4(data0.w);
		ray4.dy = make_float4(data1.x);
		ray4.dz = make_float4(data1.y);

		ray4.mint = make_float4(data1.z);
		ray4.maxt = make_float4(data1.w);
	}

	const float4 invDir0 = make_float4(1.f / ray4.dx.x);
	const float4 invDir1 = make_float4(1.f / ray4.dy.x);
	const float4 invDir2 = make_float4(1.f / ray4.dz.x);

	const int signs0 = (ray4.dx.x < 0.f);
	const int signs1 = (ray4.dy.x < 0.f);
	const int signs2 = (ray4.dz.x < 0.f);

	//RayHit rayHit;
	rayHit.index = 0xffffffffu;

	int nodeStack[QBVH_STACK_SIZE];
	nodeStack[0] = 0; // first node to handle: root node

	//------------------------------
	// Main loop
	int todoNode = 0; // the index in the stack
	// nodeStack leads to a lot of local memory banks conflicts however it has not real
	// impact on performances (I guess access latency is hiden by other stuff).
	// Avoiding conflicts is easy to do but it requires to know the work group
	// size (not worth doing if there are not performance benefits).
	//__shared__ int *nodeStack = &nodeStacks[QBVH_STACK_SIZE * threadIdx.x];
	//nodeStack[0] = 0; // first node to handle: root node

	//int maxDepth = 0;
	while (todoNode >= 0) {
		const int nodeData = nodeStack[todoNode];
		--todoNode;

		// Leaves are identified by a negative index
		if (!QBVHNode_IsLeaf(nodeData)) {
			POINTERFREESCENE::QBVHNode *node = &nodes[nodeData];
			const int4 visit = QBVHNode_BBoxIntersect(node->bboxes[signs0][0],
					node->bboxes[1 - signs0][0], node->bboxes[signs1][1],
					node->bboxes[1 - signs1][1], node->bboxes[signs2][2],
					node->bboxes[1 - signs2][2], &ray4, invDir0, invDir1,
					invDir2, signs0, signs1, signs2);

			const int4 children = node->children;

			// For some reason doing logic operations with int4 is very slow
			nodeStack[todoNode + 1] = children.w;
			todoNode += (visit.w && !QBVHNode_IsEmpty(children.w)) ? 1 : 0;
			nodeStack[todoNode + 1] = children.z;
			todoNode += (visit.z && !QBVHNode_IsEmpty(children.z)) ? 1 : 0;
			nodeStack[todoNode + 1] = children.y;
			todoNode += (visit.y && !QBVHNode_IsEmpty(children.y)) ? 1 : 0;
			nodeStack[todoNode + 1] = children.x;
			todoNode += (visit.x && !QBVHNode_IsEmpty(children.x)) ? 1 : 0;

			//maxDepth = max(maxDepth, todoNode);
		} else {
			// Perform intersection
			const uint nbQuadPrimitives = QBVHNode_NbQuadPrimitives(nodeData);
			const uint offset = QBVHNode_FirstQuadIndex(nodeData);

			for (uint primNumber = offset;
					primNumber < (offset + nbQuadPrimitives); ++primNumber) {
				POINTERFREESCENE::QuadTriangle *quadTri = &quadTris[primNumber];
				const float4 origx = quadTri->origx;
				const float4 origy = quadTri->origy;
				const float4 origz = quadTri->origz;
				const float4 edge1x = quadTri->edge1x;
				const float4 edge1y = quadTri->edge1y;
				const float4 edge1z = quadTri->edge1z;
				const float4 edge2x = quadTri->edge2x;
				const float4 edge2y = quadTri->edge2y;
				const float4 edge2z = quadTri->edge2z;
				const uint4 primitives = quadTri->primitives;
				QuadTriangle_Intersect(origx, origy, origz, edge1x, edge1y,
						edge1z, edge2x, edge2y, edge2z, primitives, &ray4,
						&rayHit);
			}
		}
	}

	//printf(\"MaxDepth=%02d\\n\", maxDepth);

	// Write result
	//		rayHit.t = rayHit.t;
	//		rayHit.b1 = rayHit.b1;
	//		rayHit.b2 = rayHit.b2;
	//		rayHits[gid].index = rayHit.index;
}

__device__ void InitPhotonPath(PhotonPath& photonPath, Ray& ray, Seed& seed) {

	//Scene *scene = ss->scene;
	// Select one light source
	float lpdf;
	float pdf;

	Spectrum f;

	//photonPath->seed = mwc();

	float u0 = getFloatRNG(seed);
	float u1 = getFloatRNG(seed);
	float u2 = getFloatRNG(seed);
	float u3 = getFloatRNG(seed);
	float u4 = getFloatRNG(seed);
	float u5 = getFloatRNG(seed);

	int lightIndex;

	POINTERFREESCENE::LightSourceType lightT = ssBuff_c->SampleAllLights(u0,
			&lpdf, lightIndex, workerBuff_c->infiniteLightBuff,
			workerBuff_c->sunLightBuff, workerBuff_c->skyLightBuff);

	if (lightT == POINTERFREESCENE::TYPE_IL_IS)
		ssBuff_c->InfiniteLight_Sample_L(u1, u2, u3, u4, u5, &pdf, &ray,
				photonPath.flux, workerBuff_c->infiniteLightBuff,
				workerBuff_c->infiniteLightMapBuff);

	else if (lightT == POINTERFREESCENE::TYPE_SUN)
		ssBuff_c->SunLight_Sample_L(u1, u2, u3, u4, u5, &pdf, &ray,
				photonPath.flux, workerBuff_c->sunLightBuff);

	else if (lightT == POINTERFREESCENE::TYPE_IL_SKY)
		ssBuff_c->SkyLight_Sample_L(u1, u2, u3, u4, u5, &pdf, &ray,
				photonPath.flux, workerBuff_c->skyLightBuff);

	else {
		ssBuff_c->TriangleLight_Sample_L(
				&workerBuff_c->areaLightsBuff[lightIndex], u1, u2, u3, u4, u5,
				&pdf, &ray, photonPath.flux, workerBuff_c->colorsBuff,
				workerBuff_c->meshDescsBuff);
	}

	//##########

	//const LightSource *light2 = scene->SampleAllLights(u0, &lpdf);

	// Initialize the photon path
	//photonPath->flux = light2->Sample_L(scene, u1, u2, u3, u4, u5, &pdf, ray);

	//#########

	photonPath.flux /= pdf * lpdf;
	photonPath.depth = 0;

	//engine->incPhotonCount();
	//atomicAdd(&photonCount, 1);
	//printf("%u\n",atomicAdd(&photonCount, 1));
}

__device__ void SavePhotonHit(Point &hitPoint, Normal &shadeN, Vector wi,
		Spectrum photonFlux) {

	unsigned long long a = atomicAdd(workerBuff_c->photonHitCountBuff, 1);

	PhotonHit* photonHits_d = workerBuff_c->photonHitsBuff;

	photonHits_d[a].hitPoint = hitPoint;
	photonHits_d[a].photonFlux = photonFlux;
	photonHits_d[a].shadeN = shadeN;
	photonHits_d[a].wi = wi;

}

__device__ bool GetHitPointInformation(Ray& ray, RayHit& rayHit,
		Point &hitPoint, Spectrum &surfaceColor, Normal &N, Normal &shadeN) {

	hitPoint = (ray)(rayHit.t);
	const unsigned int currentTriangleIndex = rayHit.index;

	unsigned int currentMeshIndex;
	unsigned int triIndex;

	currentMeshIndex = workerBuff_c->meshIDsBuff[currentTriangleIndex];
	triIndex = currentTriangleIndex
			- workerBuff_c->meshFirstTriangleOffsetBuff[currentMeshIndex];

	POINTERFREESCENE::Mesh& m =
			((POINTERFREESCENE::Mesh*) (workerBuff_c->meshDescsBuff))[currentMeshIndex];

	if (m.hasColors) {

		ssBuff_c->Mesh_InterpolateColor(
				(Spectrum*) &workerBuff_c->colorsBuff[m.colorsOffset],
				&workerBuff_c->trisBuff[m.trisOffset], triIndex, rayHit.b1,
				rayHit.b2, &surfaceColor);

	} else {
		surfaceColor = Spectrum(1.f, 1.f, 1.f);
	}

	ssBuff_c->Mesh_InterpolateNormal(&workerBuff_c->normalsBuff[m.vertsOffset],
			&(workerBuff_c->trisBuff[m.trisOffset]), triIndex, rayHit.b1,
			rayHit.b2, N);

// Flip the normal if required
	Vector& a = ray.d;
	if (Dot(a, N) > 0.f)
		shadeN = -N;
	else
		shadeN = N;

	return false;
}

__device__ void subAdvancePhotonPath(PhotonPath& photonPath, Ray& ray,
		RayHit& rayHit, Seed& seed, bool& init) {

	if (photonPath.depth >= MAX_PHOTON_PATH_DEPTH) {
		init = true;
		return;
	}

	if (rayHit.Miss()) {
		init = true;
		return;
	}

	Point hitPoint;
	Spectrum surfaceColor;
	Normal N, shadeN;

	if (GetHitPointInformation(ray, rayHit, hitPoint, surfaceColor, N, shadeN))
		return;

	const unsigned int currentTriangleIndex = rayHit.index;

	const unsigned int currentMeshIndex =
			workerBuff_c->meshIDsBuff[currentTriangleIndex];

	POINTERFREESCENE::Material *hitPointMat =
			&workerBuff_c->materialsBuff[workerBuff_c->meshMatsBuff[currentMeshIndex]];

	uint matType = hitPointMat->type;

	if (matType == MAT_AREALIGHT) {
		init = true;
		return;
	}

	bool specularBounce;

	float fPdf;
	Vector wi;
	Vector wo = -ray.d;

	float u0 = getFloatRNG(seed);
	float u1 = getFloatRNG(seed);
	float u2 = getFloatRNG(seed);

	Spectrum f;

	switch (matType) {
	case MAT_MATTE:
		ssBuff_c->Matte_Sample_f(&hitPointMat->param.matte, &wo, &wi, &fPdf, &f,
				&shadeN, u0, u1, &specularBounce);

		f *= surfaceColor;
		break;

	case MAT_MIRROR:
		ssBuff_c->Mirror_Sample_f(&hitPointMat->param.mirror, &wo, &wi, &fPdf,
				&f, &shadeN, &specularBounce);
		f *= surfaceColor;
		break;

	case MAT_GLASS:
		ssBuff_c->Glass_Sample_f(&hitPointMat->param.glass, &wo, &wi, &fPdf, &f,
				&N, &shadeN, u0, &specularBounce);
		f *= surfaceColor;

		break;

	case MAT_MATTEMIRROR:
		ssBuff_c->MatteMirror_Sample_f(&hitPointMat->param.matteMirror, &wo,
				&wi, &fPdf, &f, &shadeN, u0, u1, u2, &specularBounce);
		f *= surfaceColor;

		break;

	case MAT_METAL:
		ssBuff_c->Metal_Sample_f(&hitPointMat->param.metal, &wo, &wi, &fPdf, &f,
				&shadeN, u0, u1, &specularBounce);
		f *= surfaceColor;

		break;

	case MAT_MATTEMETAL:
		ssBuff_c->MatteMetal_Sample_f(&hitPointMat->param.matteMetal, &wo, &wi,
				&fPdf, &f, &shadeN, u0, u1, u2, &specularBounce);
		f *= surfaceColor;

		break;

	case MAT_ALLOY:
		ssBuff_c->Alloy_Sample_f(&hitPointMat->param.alloy, &wo, &wi, &fPdf, &f,
				&shadeN, u0, u1, u2, &specularBounce);
		f *= surfaceColor;

		break;

	case MAT_ARCHGLASS:
		ssBuff_c->ArchGlass_Sample_f(&hitPointMat->param.archGlass, &wo, &wi,
				&fPdf, &f, &N, &shadeN, u0, &specularBounce);
		f *= surfaceColor;

		break;

	case MAT_NULL:
		wi = ray.d;
		specularBounce = 1;
		fPdf = 1.f;
		//printf("error\n");

		break;

	default:
		// Huston, we have a problem...
		//printf("error\n");

		specularBounce = 1;
		fPdf = 0.f;
		break;
	}

// Build the next vertex path ray
	if ((fPdf <= 0.f) || f.Black()) {
		init = true;
	} else {
		photonPath.depth++;
		photonPath.flux *= f / fPdf;

		// Russian Roulette
		const float p = 0.75f;
		if (photonPath.depth < 3) {
			ray = Ray(hitPoint, wi);
		} else {
			if (getFloatRNG(seed) < p) {
				photonPath.flux /= p;
				ray = Ray(hitPoint, wi);
			} else {
				init = true;
			}
		}
	}

	if (!specularBounce) { // if difuse
		//	AddFlux(worker, engine, hashBuff, ss, engine->alpha, hitPoint, shadeN, -ray.d,
		//			photonPath.flux);

		Vector a = -ray.d;
		SavePhotonHit(hitPoint, shadeN, a, photonPath.flux);

	}

}

__global__ void initHits() {
	hits = 0;
}

__global__ void printHits() {
	printf("Photosn contributed : %u\n", hits);

}

__global__ void GenerateSeedBuffer(uint hitPointCount, uint deviceID) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= hitPointCount)
		return;

	workerBuff_c->seedsBuff[index] = mwc(index + deviceID);
	//printf("%u\n", workerBuff_c->seedsBuff[index]);

}

/**
 * possibly join with GenerateHitPointsMortonCodes
 */
__global__ void GeneratePhotonHitMortonCodes(uint hitPointsCount,
		uint* mortonCodes, uint* mortonIndex, BBox b, Vector d) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= hitPointsCount)
		return;

	PhotonHit& o1 = workerBuff_c->photonHitsBuff[index];

	mortonIndex[index] = index;

	Point p;
	p.x = (o1.hitPoint.x - b.pMin.x) * d.x;
	p.y = (o1.hitPoint.y - b.pMin.y) * d.y;
	p.z = (o1.hitPoint.z - b.pMin.z) * d.z;

	mortonCodes[index] = morton3D(p.x, p.y, p.z, MORTON_BITS);

//	printf("%.3f, %.3f, %.3f | %.3f, %.3f, %.3f | %u\n", o1.position.x,
//			o1.position.y, o1.position.z, p.x, p.y, p.z, mortonCodes[index]);

}

__global__ void GenerateHitPointsMortonCodes(HitPoint* points,
		uint hitPointsCount, uint* mortonCodes, uint* mortonIndex, BBox b,
		Vector d) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= hitPointsCount)
		return;

	Point o1 = points[index].GetPosition();
	mortonIndex[index] = index;

	if (points[index].type == CONSTANT_COLOR)
		return;

	Point p;
	p.x = (o1.x - b.pMin.x) * d.x;
	p.y = (o1.y - b.pMin.y) * d.y;
	p.z = (o1.z - b.pMin.z) * d.z;

	mortonCodes[index] = morton3D(p.x, p.y, p.z, MORTON_BITS);

//	printf("%.3f, %.3f, %.3f | %.3f, %.3f, %.3f | %u\n", o1.position.x,
//			o1.position.y, o1.position.z, p.x, p.y, p.z, mortonCodes[index]);

}

__global__ void initHashValues(uint* m_HashValue, uint* m_PointIdx, uint el,
		uint v) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= el)
		return;

	m_HashValue[index] = v;
	m_PointIdx[index] = v;

}

__global__ void GenenerateCameraRays(uint hitPointTotal, uint superSampling,
		uint width, uint height) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	unsigned int sampleIndex = 0;
	const float invSuperSampling = 1.f / superSampling;
	uint superSampling2 = superSampling * superSampling;

	for (unsigned int sy = 0; sy < superSampling; ++sy) {
		for (unsigned int sx = 0; sx < superSampling; ++sx) {

			uint hitPointsIndex = y * width * superSampling2
					+ x * superSampling2 + sampleIndex++;

			//printf("%dx%d=%d\n",x,y,hitPointsIndex);

			EyePath *eyePath = &workerBuff_c->todoEyePathsBuff[hitPointsIndex];

			eyePath->scrX =
					x
							+ (sx
									+ getFloatRNG(
											workerBuff_c->seedsBuff[hitPointsIndex]))
									* invSuperSampling - 0.5f;

			eyePath->scrY =
					y
							+ (sy
									+ getFloatRNG(
											workerBuff_c->seedsBuff[hitPointsIndex]))
									* invSuperSampling - 0.5f;

			float u0 = getFloatRNG(workerBuff_c->seedsBuff[hitPointsIndex]);
			float u1 = getFloatRNG(workerBuff_c->seedsBuff[hitPointsIndex]);
			float u2 = getFloatRNG(workerBuff_c->seedsBuff[hitPointsIndex]);

			ssBuff_c->GenerateRay(eyePath->scrX, eyePath->scrY, width, height,
					&eyePath->ray, u0, u1, u2, &ssBuff_c->camera);

			eyePath->depth = 0;
			eyePath->throughput = Spectrum(1.f, 1.f, 1.f);

			eyePath->done = false;
			eyePath->splat = false;
			eyePath->sampleIndex = hitPointsIndex;

			HitPoint* hp =
					&workerBuff_c->workerHitPointsInfoBuff[hitPointsIndex];

			hp->id = hitPointsIndex;

		}
	}
}

/**
 * A thread per raybuffer/workbuffer entry, when path finished initializes another
 */

__global__ void
__launch_bounds__( PHOTONPASS_MAX_THREADS_PER_BLOCK)
fullAdvance(uint photonTarget) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < photonTarget) {

		Ray ray;
		RayHit rayHit;
		Seed& seed = workerBuff_c->seedsBuff[tid];
		PhotonPath photonPath;

		InitPhotonPath(photonPath, ray, seed);

		bool done = false;

		while (!done) {

			subIntersect(ray, workerBuff_c->d_qbvhBuff,
					workerBuff_c->d_qbvhTrisBuff, rayHit);

			subAdvancePhotonPath(photonPath, ray, rayHit,
					workerBuff_c->seedsBuff[tid], done);

		}

		atomicAdd(workerBuff_c->photonCountBuff, 1);

	}

}

__global__ void Intersect(Ray *rays, RayHit *rayHits,
		POINTERFREESCENE::QBVHNode *nodes,
		POINTERFREESCENE::QuadTriangle *quadTris, const uint rayCount) {

//	// Select the ray to check
//	int len_X = gridDim.x * blockDim.x;
//	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
//	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	int gid = pos_y * len_X + pos_x;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < rayCount) {

		// Prepare the ray for intersection
		POINTERFREESCENE::QuadRay ray4;
		{
			float4 *basePtr = (float4 *) &rays[gid];
			float4 data0 = (*basePtr++);
			float4 data1 = (*basePtr);

			ray4.ox = make_float4(data0.x);
			ray4.oy = make_float4(data0.y);
			ray4.oz = make_float4(data0.z);

			ray4.dx = make_float4(data0.w);
			ray4.dy = make_float4(data1.x);
			ray4.dz = make_float4(data1.y);

			ray4.mint = make_float4(data1.z);
			ray4.maxt = make_float4(data1.w);
		}

		const float4 invDir0 = make_float4(1.f / ray4.dx.x);
		const float4 invDir1 = make_float4(1.f / ray4.dy.x);
		const float4 invDir2 = make_float4(1.f / ray4.dz.x);

		const int signs0 = (ray4.dx.x < 0.f);
		const int signs1 = (ray4.dy.x < 0.f);
		const int signs2 = (ray4.dz.x < 0.f);

		RayHit rayHit;
		rayHit.index = 0xffffffffu;

		int nodeStack[QBVH_STACK_SIZE];
		nodeStack[0] = 0; // first node to handle: root node

		//------------------------------
		// Main loop
		int todoNode = 0; // the index in the stack
		// nodeStack leads to a lot of local memory banks conflicts however it has not real
		// impact on performances (I guess access latency is hiden by other stuff).
		// Avoiding conflicts is easy to do but it requires to know the work group
		// size (not worth doing if there are not performance benefits).
		//__shared__ int *nodeStack = &nodeStacks[QBVH_STACK_SIZE * threadIdx.x];
		//nodeStack[0] = 0; // first node to handle: root node

		//int maxDepth = 0;
		while (todoNode >= 0) {
			const int nodeData = nodeStack[todoNode];
			--todoNode;

			// Leaves are identified by a negative index
			if (!QBVHNode_IsLeaf(nodeData)) {
				POINTERFREESCENE::QBVHNode *node = &nodes[nodeData];
				const int4 visit = QBVHNode_BBoxIntersect(
						node->bboxes[signs0][0], node->bboxes[1 - signs0][0],
						node->bboxes[signs1][1], node->bboxes[1 - signs1][1],
						node->bboxes[signs2][2], node->bboxes[1 - signs2][2],
						&ray4, invDir0, invDir1, invDir2, signs0, signs1,
						signs2);

				const int4 children = node->children;

				// For some reason doing logic operations with int4 is very slow
				nodeStack[todoNode + 1] = children.w;
				todoNode += (visit.w && !QBVHNode_IsEmpty(children.w)) ? 1 : 0;
				nodeStack[todoNode + 1] = children.z;
				todoNode += (visit.z && !QBVHNode_IsEmpty(children.z)) ? 1 : 0;
				nodeStack[todoNode + 1] = children.y;
				todoNode += (visit.y && !QBVHNode_IsEmpty(children.y)) ? 1 : 0;
				nodeStack[todoNode + 1] = children.x;
				todoNode += (visit.x && !QBVHNode_IsEmpty(children.x)) ? 1 : 0;

				//maxDepth = max(maxDepth, todoNode);
			} else {
				// Perform intersection
				const uint nbQuadPrimitives = QBVHNode_NbQuadPrimitives(
						nodeData);
				const uint offset = QBVHNode_FirstQuadIndex(nodeData);

				for (uint primNumber = offset;
						primNumber < (offset + nbQuadPrimitives);
						++primNumber) {
					POINTERFREESCENE::QuadTriangle *quadTri =
							&quadTris[primNumber];
					const float4 origx = quadTri->origx;
					const float4 origy = quadTri->origy;
					const float4 origz = quadTri->origz;
					const float4 edge1x = quadTri->edge1x;
					const float4 edge1y = quadTri->edge1y;
					const float4 edge1z = quadTri->edge1z;
					const float4 edge2x = quadTri->edge2x;
					const float4 edge2y = quadTri->edge2y;
					const float4 edge2z = quadTri->edge2z;
					const uint4 primitives = quadTri->primitives;
					QuadTriangle_Intersect(origx, origy, origz, edge1x, edge1y,
							edge1z, edge2x, edge2y, edge2z, primitives, &ray4,
							&rayHit);
				}
			}
		}

		//printf(\"MaxDepth=%02d\\n\", maxDepth);

		// Write result
		rayHits[gid].t = rayHit.t;
		rayHits[gid].b1 = rayHit.b1;
		rayHits[gid].b2 = rayHit.b2;
		rayHits[gid].index = rayHit.index;

		//printf("rayHits[%d].index = %u,t = %.4f\n",gid,rayHits[gid].index,rayHits[gid].t);

	}
}

__global__ void AccumulateFluxPPM(u_int64_t photonTraced, uint hitPointTotal,
		float alpha) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < hitPointTotal) {

		HitPoint *ihp = &workerBuff_c->workerHitPointsInfoBuff[tid];
		//HitPointRadianceFlux *ihp = &worker->workerHitPointsBuff[tid];

		switch (ihp->type) {
		case CONSTANT_COLOR:
			ihp->radiance = ihp->throughput;
			break;
		case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				const unsigned long long pcount = ihp->photonCount
						+ ihp->accumPhotonCount;

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

}

__global__ void AccumulateFluxSPPM(u_int64_t photonTraced, uint hitPointTotal,
		float alpha) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < hitPointTotal) {

		HitPoint *ihp = &workerBuff_c->workerHitPointsInfoBuff[tid];
		//HitPointRadianceFlux *ihp = &worker->workerHitPointsBuff[tid];

		switch (ihp->type) {
		case CONSTANT_COLOR:
			ihp->accumRadiance += ihp->throughput;
			ihp->constantHitsCount += 1;
			break;

		case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				const unsigned long long pcount = ihp->photonCount
						+ ihp->accumPhotonCount;

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
}

__global__ void AccumulateFluxPPMPA(float currentPhotonRadius2,
		u_int64_t photonTraced, uint hitPointTotal) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < hitPointTotal) {

		HitPoint *ihp = &workerBuff_c->workerHitPointsInfoBuff[tid];
		//HitPointRadianceFlux *ihp = &worker->workerHitPointsBuff[tid];

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

__global__ void AccumulateFluxSPPMPA(float currentPhotonRadius2,
		u_int64_t photonTraced, uint hitPointTotal) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < hitPointTotal) {

		HitPoint *ihp = &workerBuff_c->workerHitPointsInfoBuff[tid];
		//HitPointRadianceFlux *ihp = &worker->workerHitPointsBuff[tid];

		switch (ihp->type) {
		case CONSTANT_COLOR:
			ihp->accumRadiance = ihp->throughput;
			ihp->constantHitsCount = 1;
			break;
		case SURFACE:

			if ((ihp->accumPhotonCount > 0)) {

				ihp->reflectedFlux = ihp->accumReflectedFlux;
				ihp->accumPhotonCount = 0;
				ihp->accumReflectedFlux = Spectrum();

			}

			ihp->surfaceHitsCount = 1;
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

}

__device__ int3 GetGridPos(float4 p_Pos) {
	int3 gridPos;
	gridPos.x = floorf((p_Pos.x - g_Params.bbMin.x) * g_Params.invCellSize.x);
	gridPos.y = floorf((p_Pos.y - g_Params.bbMin.y) * g_Params.invCellSize.y);
	gridPos.z = floorf((p_Pos.z - g_Params.bbMin.z) * g_Params.invCellSize.z);
//
	if (gridPos.x < 0 || gridPos.y < 0 || gridPos.z < 0)
		gridPos.x = 0;

	return gridPos;
}

__device__ uint CalcGridHash(int3 p_GridPos) {

	const uint p1 = 73856093; // some large primes
	const uint p2 = 19349663;
	const uint p3 = 83492791;
	uint n = __umul24(p_GridPos.x, p1) ^ __umul24(p_GridPos.y, p2)
			^ __umul24(p_GridPos.z, p3);
	return (n & (g_Params.SpatialHashTableSize - 1)); // % SpatialHashTableSize if a multiple of 2

}

__global__ void CalcPositionHashes(HitPoint* hitpoints, uint* m_HashValue,
		uint* m_PointIdx, uint p_Elements, float currentPhotonRadius2,
		Point hpBBoxpMin, float invCellSize) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= p_Elements)
		return;

	HitPoint *hpp = &hitpoints[index];

	if (hpp->type == CONSTANT_COLOR)
		return;

//photonRadius = sqrtf(hpp->accumPhotonRadius2);

	float photonRadius;
	if (g_Params.eT == PPMPA || g_Params.eT == SPPMPA)
		photonRadius = sqrtf(currentPhotonRadius2);

	else {
		photonRadius = sqrtf(hpp->accumPhotonRadius2);
	}

	const Vector rad(photonRadius, photonRadius, photonRadius);

	const Vector bMin = ((hpp->position - rad) - hpBBoxpMin) * invCellSize;
	const Vector bMax = ((hpp->position + rad) - hpBBoxpMin) * invCellSize;

//	if (bMin.x < -71039.8662 || bMin.x > 71039.8662)
//		printf("%d ", hp->id);

	uint cell = 0;
	for (int iz = abs(int(bMin.z)); iz <= abs(int(bMax.z)); iz++) {
		for (int iy = abs(int(bMin.y)); iy <= abs(int(bMax.y)); iy++) {
			for (int ix = abs(int(bMin.x)); ix <= abs(int(bMax.x)); ix++) {

				//int3 gridPos = GetGridPos(make_float4(ix, iy, iz, 0.f));

				uint hashValue;
				hashValue = CalcGridHash(make_int3(ix, iy, iz));

				// write result into output list
				m_HashValue[(index * 8) + cell] = hashValue;
				m_PointIdx[(index * 8) + cell++] = index;
			}
		}
	}
}

__global__ void CalcPositionMortonHashes(HitPoint* hitpoints, uint* m_HashValue,
		uint* m_PointIdx, uint p_Elements, float currentPhotonRadius2,
		Point hpBBoxpMin, float invCellSize, Vector totalCells) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= p_Elements)
		return;

	HitPoint *hpp = &hitpoints[index];

	m_PointIdx[index] = index;

	if (hpp->type == CONSTANT_COLOR)
		return;

//	float photonRadius;
//	if (g_Params.eT == PPMPA || g_Params.eT == SPPMPA)
//		photonRadius = sqrtf(currentPhotonRadius2);
//
//	else {
//		photonRadius = sqrtf(hpp->accumPhotonRadius2);
//	}

//	const Vector rad(photonRadius, photonRadius, photonRadius);
//
//	const Vector bMin = ((hpp->position - rad) - hpBBoxpMin) * invCellSize;
//	const Vector bMax = ((hpp->position + rad) - hpBBoxpMin) * invCellSize;
//
////	if (bMin.x < -71039.8662 || bMin.x > 71039.8662)
////		printf("%d ", hp->id);
//
//	uint cell = 0;
//	for (int iz = abs(int(bMin.z)); iz <= abs(int(bMax.z)); iz++) {
//		for (int iy = abs(int(bMin.y)); iy <= abs(int(bMax.y)); iy++) {
//			for (int ix = abs(int(bMin.x)); ix <= abs(int(bMax.x)); ix++) {

	int3 gridPos = GetGridPos(
			make_float4(hpp->position.x, hpp->position.y, hpp->position.z,
					0.f));

	uint hashValue;

	Point p(gridPos.x / totalCells.x, gridPos.y / totalCells.y,
			gridPos.z / totalCells.z);

//				assert(p.x >= 0 && p.x <= 1);
//				assert(p.y >= 0 && p.y <= 1);
//				assert(p.z >= 0 && p.z <= 1);
	hashValue = morton3D(p.x, p.y, p.z, MORTON_BITS);

	//hashValue = CalcGridHash(gridPos);


	// write result into output list
	m_HashValue[index] = hashValue;

//			}
//		}
//	}
}

__device__ int BinarySearchHP(const HitPoint* p_List, uint p_Size, uint p_Key) {
	int left, right, midpt;
	left = 0;
	right = p_Size - 1;
	while (left <= right) {
		midpt = (left + right) >> 1;
		if (p_Key == p_List[midpt].id)
			return static_cast<int>(midpt);
		else if (p_Key > p_List[midpt].id)
			left = midpt + 1;
		else
			right = midpt - 1;
	}

	return -1;
}

__device__ int BinarySearch(const uint* p_List, uint p_Size, uint p_Key) {
	int left, right, midpt;
	left = 0;
	right = p_Size - 1;
	while (left <= right) {
		midpt = (left + right) >> 1;
		if (p_Key == p_List[midpt])
			return static_cast<int>(midpt);
		else if (p_Key > p_List[midpt])
			left = midpt + 1;
		else
			right = midpt - 1;
	}

	return -1;
}

__global__ void CreateHashTable(const uint* p_Hashes, int* m_FirstIdx,
		uint* m_NumPhotons, uint numHashes, uint cellCount) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= cellCount)
		return;

// perform binary search to find elements with our hash in the sorted hash list
	int pos = BinarySearch(p_Hashes, numHashes, index);

// locate beginning of hash sequence
	int originalPos = pos;
	if (pos > 0) {
		while (pos >= 0 && p_Hashes[pos] == index)
			--pos;
		++pos;

		// find end of the sequence
		++originalPos;
		while (originalPos < (numHashes) && p_Hashes[originalPos] == index)
			++originalPos;
	}

// store result in table
	m_FirstIdx[index] = pos;
	m_NumPhotons[index] = originalPos - pos;

}

__global__ void PhotonSearchHash(int* FirstIdxBuff, uint* NumPhotonsBuff,
		uint* PointIdx, float currentPhotonRadius2, uint p_NumPoints,
		engineType eT, uint hitPointTotal) {

	uint qIndex = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (qIndex < p_NumPoints) {

#warning HACK

		PhotonHit* photonHit = &workerBuff_c->photonHitsBuff[qIndex];

		float4 curPos = (make_float4(*(float3*) &photonHit->hitPoint));

		float PhotonRadius2;
//		for (int curCell = 0; curCell < 27; ++curCell) {
//			// calc cells position
//			float4 offsetPos = curPos + g_CellOffsets[curCell];
//
		// get cell hash

		int3 gridPos = GetGridPos(curPos);
		uint hashValue;

//		Point p(gridPos.x / totalCells.x, gridPos.y / totalCells.y,
//				gridPos.z / totalCells.z);
//
////		assert(p.x >= 0 && p.x <= 1);
////		assert(p.y >= 0 && p.y <= 1);
////		assert(p.z >= 0 && p.z <= 1);
//
//		hashValue = morton3D(p.x, p.y, p.z, MORTON_BITS);
//
//		assert(hashValue < (1 << (3 * MORTON_BITS)));

		hashValue = CalcGridHash(gridPos);

		// process photons

		int curHitpoint = FirstIdxBuff[hashValue];

		uint numPhotons = NumPhotonsBuff[hashValue];

		for (uint i = 0; i < numPhotons; ++i) {
			// calc photon distance

			uint hitpoint = PointIdx[curHitpoint];
			//uint hitpoint = curHitpoint;

			assert(hitpoint < hitPointTotal);

			HitPoint *ihp = &workerBuff_c->workerHitPointsInfoBuff[hitpoint];

//			if (ihp->type == CONSTANT_COLOR)
//							continue;

//			HitPointRadianceFlux * hp =
//					&workerBuff_c->workerHitPointsBuff[hitpoint];

#warning HACK
			//float4 distVec = hitpointsBuff[curPhoton].position - curPos;
			Vector dist = ihp->position - photonHit->hitPoint;

			//	float4 distVec = (make_float4(*(float3*) &ihp->position))
			//					- curPos;
			//float v = distVec.x * distVec.x + distVec.y * distVec.y
			//	+ distVec.z * distVec.z;

			//			if (curSqDist < curMaxsqDist && curSqDist < p_MaxSqDist) {

			if (eT == PPM || eT == SPPM)
				PhotonRadius2 = ihp->accumPhotonRadius2;
			else
				PhotonRadius2 = currentPhotonRadius2;

			if ((Dot(ihp->normal, photonHit->shadeN) > 0.5f)
					&& Dot(dist, dist) <= PhotonRadius2) {

				atomicAdd(&ihp->accumPhotonCount, 1);
				atomicAdd(&hits, 1);

				Spectrum f;

				POINTERFREESCENE::Material *hitPointMat =
						&workerBuff_c->materialsBuff[ihp->materialSS];

				switch (hitPointMat->type) {

				case MAT_MATTE:
					ssBuff_c->Matte_f(&hitPointMat->param.matte, ihp->wo,
							photonHit->wi, photonHit->shadeN, f);
					break;

				case MAT_MATTEMIRROR:
					ssBuff_c->MatteMirror_f(&hitPointMat->param.matteMirror,
							ihp->wo, photonHit->wi, photonHit->shadeN, f);
					break;

				case MAT_MATTEMETAL:
					ssBuff_c->MatteMetal_f(&hitPointMat->param.matteMetal,
							ihp->wo, photonHit->wi, photonHit->shadeN, f);
					break;

				case MAT_ALLOY:
					ssBuff_c->Alloy_f(&hitPointMat->param.alloy, ihp->wo,
							photonHit->wi, photonHit->shadeN, f);

					break;
				default:

					break;

				}

				Spectrum flux = photonHit->photonFlux
						* AbsDot(photonHit->shadeN, photonHit->wi)
						* ihp->throughput * (f);

				atomicAdd(&ihp->accumReflectedFlux.r, flux.r);
				atomicAdd(&ihp->accumReflectedFlux.g, flux.g);
				atomicAdd(&ihp->accumReflectedFlux.b, flux.b);

			}
			++curHitpoint;
		}
		//}

	}
}

__constant__ float4 g_CellOffsets[] = { { 0, 0, 0, 0 },

{ 1, 0, 0, 0 }, { 1, 0, 1, 0 }, { 1, 0, -1, 0 },

{ 1, 1, 0, 0 }, { 1, 1, 1, 0 }, { 1, 1, -1, 0 },

{ 1, -1, 0, 0 }, { 1, -1, 1, 0 }, { 1, -1, -1, 0 },

{ -1, 0, 0, 0 }, { -1, 0, 1, 0 }, { -1, 0, -1, 0 },

{ -1, 1, 0, 0 }, { -1, 1, 1, 0 }, { -1, 1, -1, 0 },

{ -1, -1, 0, 0 }, { -1, -1, 1, 0 }, { -1, -1, -1, 0 },

{ 0, 0, 1, 0 }, { 0, -1, 1, 0 }, { 0, 1, 1, 0 },

{ 0, 0, -1, 0 }, { 0, -1, -1, 0 }, { 0, 1, -1, 0 },

{ 0, 1, 0, 0 }, { 0, -1, 0, 0 } };

__global__ void PhotonSearchMortonHash(int* FirstIdxBuff, uint* NumPhotonsBuff,
		float currentPhotonRadius2, uint p_NumPoints,
		engineType eT, uint hitPointTotal, Vector totalCells) {

	uint qIndex = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (qIndex < p_NumPoints) {

#warning HACK

		PhotonHit* photonHit = &workerBuff_c->photonHitsBuff[qIndex];

		float4 curPos = (make_float4(*(float3*) &photonHit->hitPoint));

		float PhotonRadius2;
		for (int curCell = 0; curCell < 27; ++curCell) {
//			// calc cells position
			float4 offsetPos = curPos + g_CellOffsets[curCell];

		// get cell hash

		int3 gridPos = GetGridPos(offsetPos);
		uint hashValue;

		Point p(gridPos.x / totalCells.x, gridPos.y / totalCells.y,
				gridPos.z / totalCells.z);

//		assert(p.x >= 0 && p.x <= 1);
//		assert(p.y >= 0 && p.y <= 1);
//		assert(p.z >= 0 && p.z <= 1);

		hashValue = morton3D(p.x, p.y, p.z, MORTON_BITS);

		// hashValue = CalcGridHash(gridPos);


		assert(hashValue < (1 << (3 * MORTON_BITS)));

		// process photons

		int curHitpoint = FirstIdxBuff[hashValue];

		uint numPhotons = NumPhotonsBuff[hashValue];

		for (uint i = 0; i < numPhotons; ++i) {

			uint hitpoint = curHitpoint;

			assert(hitpoint < hitPointTotal);

			HitPoint *ihp = &workerBuff_c->workerHitPointsInfoBuff[hitpoint];

//			if (ihp->type == CONSTANT_COLOR)
//							continue;

//			HitPointRadianceFlux * hp =
//					&workerBuff_c->workerHitPointsBuff[hitpoint];

#warning HACK
			//float4 distVec = hitpointsBuff[curPhoton].position - curPos;
			Vector dist = ihp->position - photonHit->hitPoint;

			//	float4 distVec = (make_float4(*(float3*) &ihp->position))
			//					- curPos;
			//float v = distVec.x * distVec.x + distVec.y * distVec.y
			//	+ distVec.z * distVec.z;

			//			if (curSqDist < curMaxsqDist && curSqDist < p_MaxSqDist) {

			if (eT == PPM || eT == SPPM)
				PhotonRadius2 = ihp->accumPhotonRadius2;
			else
				PhotonRadius2 = currentPhotonRadius2;

			if ((Dot(ihp->normal, photonHit->shadeN) > 0.5f)
					&& Dot(dist, dist) <= PhotonRadius2) {

				atomicAdd(&ihp->accumPhotonCount, 1);
				atomicAdd(&hits, 1);

				Spectrum f;

				POINTERFREESCENE::Material *hitPointMat =
						&workerBuff_c->materialsBuff[ihp->materialSS];

				switch (hitPointMat->type) {

				case MAT_MATTE:
					ssBuff_c->Matte_f(&hitPointMat->param.matte, ihp->wo,
							photonHit->wi, photonHit->shadeN, f);
					break;

				case MAT_MATTEMIRROR:
					ssBuff_c->MatteMirror_f(&hitPointMat->param.matteMirror,
							ihp->wo, photonHit->wi, photonHit->shadeN, f);
					break;

				case MAT_MATTEMETAL:
					ssBuff_c->MatteMetal_f(&hitPointMat->param.matteMetal,
							ihp->wo, photonHit->wi, photonHit->shadeN, f);
					break;

				case MAT_ALLOY:
					ssBuff_c->Alloy_f(&hitPointMat->param.alloy, ihp->wo,
							photonHit->wi, photonHit->shadeN, f);

					break;
				default:

					break;

				}

				Spectrum flux = photonHit->photonFlux
						* AbsDot(photonHit->shadeN, photonHit->wi)
						* ihp->throughput * (f);

				atomicAdd(&ihp->accumReflectedFlux.r, flux.r);
				atomicAdd(&ihp->accumReflectedFlux.g, flux.g);
				atomicAdd(&ihp->accumReflectedFlux.b, flux.b);

			}
			++curHitpoint;
		}
		}

	}
}

__global__ void PhotonSearchMorton(int* FirstIdxBuff, uint* NumPhotonsBuff,
		float currentPhotonRadius2, uint p_NumPoints, engineType eT,
		uint hitPointTotal, Vector invD, Point pMin, float cellSize,
		int nboxes) {

	uint qIndex = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (qIndex < p_NumPoints) {

		float PhotonRadius2;

		float photonRadius = sqrt(currentPhotonRadius2);

		PhotonHit* photonHit = &workerBuff_c->photonHitsBuff[qIndex];

		const Vector rad(photonRadius, photonRadius, photonRadius);

		Point photonHitPOS = photonHit->hitPoint;
//
//		Point photonHitD = photonHit->hitPoint + rad;
//
//		photonHitD.x = (photonHitD.x - pMin.x) * invD.x;
//		photonHitD.y = (photonHitD.y - pMin.y) * invD.y;
//		photonHitD.z = (photonHitD.z - pMin.z) * invD.z;
//
		photonHitPOS.x = (photonHitPOS.x - pMin.x) * invD.x;
		photonHitPOS.y = (photonHitPOS.y - pMin.y) * invD.y;
		photonHitPOS.z = (photonHitPOS.z - pMin.z) * invD.z;
//
//		Vector r = photonHitD - photonHitPOS;
//		float dist = Dot(r, r);

//		int nboxes = (int)ceil(sqrt(dist) / cellSize);

		for (int i = -nboxes; i < nboxes; ++i) {
			for (int j = -nboxes; j < nboxes; ++j) {
				for (int k = -nboxes; k < nboxes; ++k) {

					uint cell_ijk = morton3D(photonHitPOS.x + i * cellSize,
							photonHitPOS.y + j * cellSize,
							photonHitPOS.z + k * cellSize, MORTON_BITS);

					assert(cell_ijk < (1 << (3 * MORTON_BITS)));

					int curHitpoint = FirstIdxBuff[cell_ijk];

					uint numPhotons = NumPhotonsBuff[cell_ijk];

					for (uint i = 0; i < numPhotons; ++i) {
						// calc photon distance

						uint hitpoint = curHitpoint;

						assert(hitpoint < hitPointTotal);

						HitPoint *ihp =
								&workerBuff_c->workerHitPointsInfoBuff[hitpoint];

						assert(ihp->type != CONSTANT_COLOR);

#warning HACK
						//float4 distVec = hitpointsBuff[curPhoton].position - curPos;
						Vector dist = ihp->position - photonHit->hitPoint;

						//	float4 distVec = (make_float4(*(float3*) &ihp->position))
						//					- curPos;
						//float v = distVec.x * distVec.x + distVec.y * distVec.y
						//	+ distVec.z * distVec.z;

						//			if (curSqDist < curMaxsqDist && curSqDist < p_MaxSqDist) {

						if (eT == PPM || eT == SPPM)
							PhotonRadius2 = ihp->accumPhotonRadius2;
						else
							PhotonRadius2 = currentPhotonRadius2;

						if ((Dot(ihp->normal, photonHit->shadeN) > 0.5f)
								&& Dot(dist, dist) <= PhotonRadius2) {

							atomicAdd(&ihp->accumPhotonCount, 1);
							atomicAdd(&hits, 1);

							Spectrum f;

							POINTERFREESCENE::Material *hitPointMat =
									&workerBuff_c->materialsBuff[ihp->materialSS];

							switch (hitPointMat->type) {

							case MAT_MATTE:
								ssBuff_c->Matte_f(&hitPointMat->param.matte,
										ihp->wo, photonHit->wi,
										photonHit->shadeN, f);
								break;

							case MAT_MATTEMIRROR:
								ssBuff_c->MatteMirror_f(
										&hitPointMat->param.matteMirror,
										ihp->wo, photonHit->wi,
										photonHit->shadeN, f);
								break;

							case MAT_MATTEMETAL:
								ssBuff_c->MatteMetal_f(
										&hitPointMat->param.matteMetal, ihp->wo,
										photonHit->wi, photonHit->shadeN, f);
								break;

							case MAT_ALLOY:
								ssBuff_c->Alloy_f(&hitPointMat->param.alloy,
										ihp->wo, photonHit->wi,
										photonHit->shadeN, f);

								break;
							default:

								break;

							}

							Spectrum flux = photonHit->photonFlux
									* AbsDot(photonHit->shadeN, photonHit->wi)
									* ihp->throughput * (f);

							atomicAdd(&ihp->accumReflectedFlux.r, flux.r);
							atomicAdd(&ihp->accumReflectedFlux.g, flux.g);
							atomicAdd(&ihp->accumReflectedFlux.b, flux.b);

						}
						++curHitpoint;
					}
				}
			}
		}

	}
}

__device__ void AdvanceEyePaths(bool& done, uint index, RayHit* rayHit,
		EyePath* eyePath) {

	HitPoint *hp = &workerBuff_c->workerHitPointsInfoBuff[index];

	if (rayHit->Miss()) {

		hp->type = CONSTANT_COLOR;
		hp->scrX = eyePath->scrX;
		hp->scrY = eyePath->scrY;

		if (workerBuff_c->infiniteLightBuff || workerBuff_c->sunLightBuff
				|| workerBuff_c->skyLightBuff) {

			if (workerBuff_c->infiniteLightBuff)
				ssBuff_c->InfiniteLight_Le(&(hp->throughput),
						(Vector*) &eyePath->ray.d, ssBuff_c->infiniteLight,
						ssBuff_c->infiniteLightMap);
			if (workerBuff_c->sunLightBuff)
				ssBuff_c->SunLight_Le(&hp->throughput,
						(Vector*) &eyePath->ray.d, ssBuff_c->sunLight);
			if (workerBuff_c->skyLightBuff)
				ssBuff_c->SkyLight_Le(&hp->throughput,
						(Vector*) &eyePath->ray.d, ssBuff_c->skyLight);

			hp->throughput *= eyePath->throughput;
		} else
			hp->throughput = Spectrum();

		done = true;

	} else {

		// Something was hit
		Point hitPoint;
		Spectrum surfaceColor;
		Normal N, shadeN;

		//if (rayHit->index > )

		if (GetHitPointInformation(eyePath->ray, *rayHit, hitPoint,
				surfaceColor, N, shadeN))
			return;

		// Get the material
		const unsigned int currentTriangleIndex = rayHit->index;

		const unsigned int currentMeshIndex =
				workerBuff_c->meshIDsBuff[currentTriangleIndex];

		const uint materialIndex = workerBuff_c->meshMatsBuff[currentMeshIndex];

		POINTERFREESCENE::Material *hitPointMat =
				&workerBuff_c->materialsBuff[materialIndex];

		uint matType = hitPointMat->type;

		if (matType == MAT_AREALIGHT) {
			// Add an hit point
			hp->type = CONSTANT_COLOR;
			hp->scrX = eyePath->scrX;
			hp->scrY = eyePath->scrY;
			Vector d = -eyePath->ray.d;
			ssBuff_c->AreaLight_Le(&hitPointMat->param.areaLight, &d, &N,
					&hp->throughput);
			hp->throughput *= eyePath->throughput;

			// Free the eye path
			done = true;

		} else {
			//done = true;

			Vector wo = -eyePath->ray.d;
			float materialPdf;

			Vector wi;
			bool specularMaterial = true;
			float u0 = getFloatRNG(workerBuff_c->seedsBuff[index]);
			float u1 = getFloatRNG(workerBuff_c->seedsBuff[index]);
			float u2 = getFloatRNG(workerBuff_c->seedsBuff[index]);
			Spectrum f;

			switch (matType) {

			case MAT_MATTE:
				ssBuff_c->Matte_Sample_f(&hitPointMat->param.matte, &wo, &wi,
						&materialPdf, &f, &shadeN, u0, u1, &specularMaterial);
				f *= surfaceColor;
				break;

			case MAT_MIRROR:
				ssBuff_c->Mirror_Sample_f(&hitPointMat->param.mirror, &wo, &wi,
						&materialPdf, &f, &shadeN, &specularMaterial);
				f *= surfaceColor;
				break;

			case MAT_GLASS:
				ssBuff_c->Glass_Sample_f(&hitPointMat->param.glass, &wo, &wi,
						&materialPdf, &f, &N, &shadeN, u0, &specularMaterial);
				f *= surfaceColor;

				break;

			case MAT_MATTEMIRROR:
				ssBuff_c->MatteMirror_Sample_f(&hitPointMat->param.matteMirror,
						&wo, &wi, &materialPdf, &f, &shadeN, u0, u1, u2,
						&specularMaterial);
				f *= surfaceColor;

				break;

			case MAT_METAL:
				ssBuff_c->Metal_Sample_f(&hitPointMat->param.metal, &wo, &wi,
						&materialPdf, &f, &shadeN, u0, u1, &specularMaterial);
				f *= surfaceColor;

				break;

			case MAT_MATTEMETAL:
				ssBuff_c->MatteMetal_Sample_f(&hitPointMat->param.matteMetal,
						&wo, &wi, &materialPdf, &f, &shadeN, u0, u1, u2,
						&specularMaterial);
				f *= surfaceColor;

				break;

			case MAT_ALLOY:
				ssBuff_c->Alloy_Sample_f(&hitPointMat->param.alloy, &wo, &wi,
						&materialPdf, &f, &shadeN, u0, u1, u2,
						&specularMaterial);
				f *= surfaceColor;

				break;

			case MAT_ARCHGLASS:
				ssBuff_c->ArchGlass_Sample_f(&hitPointMat->param.archGlass, &wo,
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
				hp->type = CONSTANT_COLOR;
				hp->scrX = eyePath->scrX;
				hp->scrY = eyePath->scrY;
				hp->throughput = Spectrum();
				done = true;

			} else if (specularMaterial || (!hitPointMat->difuse)) {

				eyePath->throughput *= f / materialPdf;
				eyePath->ray = Ray(hitPoint, wi);

			} else {
				// Add an hit point
				hp->type = SURFACE;
				hp->scrX = eyePath->scrX;
				hp->scrY = eyePath->scrY;
				hp->materialSS = materialIndex;
				hp->throughput = eyePath->throughput * surfaceColor;
				hp->position = hitPoint;
				hp->wo = -eyePath->ray.d;
				hp->normal = shadeN;

				// Free the eye path
				done = true;

			}

		}

		if (eyePath->depth > MAX_EYE_PATH_DEPTH) {
//
//			// Add an hit point
//			HitPointPositionInfo* hp =
//					&workerBuff_c->workerHitPointsInfoBuff[tid];

			hp->type = CONSTANT_COLOR;
			hp->scrX = eyePath->scrX;
			hp->scrY = eyePath->scrY;
			hp->throughput = Spectrum();

			done = true;

		} else if (!done) {
			eyePath->depth++;

		}

	}

}


__global__
__launch_bounds__( EYEPASS_MAX_THREADS_PER_BLOCK)
void fullAdvanceHitpoints(uint hitpointCount,
		unsigned long long* rayTraceCount) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < hitpointCount) {

		RayHit rayHit;
		EyePath eyePath = workerBuff_c->todoEyePathsBuff[tid];

		bool done = false;

		while (!done) {

			subIntersect(eyePath.ray, workerBuff_c->d_qbvhBuff,
					workerBuff_c->d_qbvhTrisBuff, rayHit);

			atomicAdd(rayTraceCount, 1);

			AdvanceEyePaths(done, tid, &rayHit, &eyePath);

		}

	}

}

__global__ void HitPointToSample(uint hitPointTotal) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= hitPointTotal)
		return;

	SampleBufferElem * s = &workerBuff_c->sampleBufferBuff[index];

	HitPoint *ihp = &workerBuff_c->workerHitPointsInfoBuff[index];

	s->screenX = ihp->scrX;
	s->screenY = ihp->scrY;
	s->radiance = ihp->radiance;
	s->id = ihp->id;

}

template<class T>
__global__ void ReorderPoints(T* points, T* old_order, uint* p_NewOrder,
		uint p_Elements) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= p_Elements)
		return;

	uint oldIndex = p_NewOrder[index];
	points[index] = old_order[oldIndex];

}
