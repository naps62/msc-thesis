#include "core.h"
#include "omp.h"
#include "random.h"
#include "atomic.h"
#include "hitpoints.h"
#include "cppbench.h"
#include "cuda_utils.h"
#include "renderEngine.h"

Engine::~Engine() {
	delete film;
	delete ss;
}

PPMEngine::~PPMEngine() {
}

bool Engine::GetHitPointInformation(PointerFreeScene *ss, Ray *ray,
		const RayHit *rayHit, Point &hitPoint, Spectrum &surfaceColor,
		Normal &N, Normal &shadeN) {

	hitPoint = (*ray)(rayHit->t);
	const unsigned int currentTriangleIndex = rayHit->index;

	unsigned int currentMeshIndex;
	unsigned int triIndex;

	//########

	//currentMeshIndex = scene->dataSet->GetMeshID(currentTriangleIndex);

	// Get the triangle
	//const ExtMesh *mesh = scene->objects[currentMeshIndex];

	//triIndex = scene->dataSet->GetMeshTriangleID(currentTriangleIndex);

	//		if (mesh->HasColors())
	//			surfaceColor = mesh->InterpolateTriColor(triIndex, rayHit->b1,
	//					rayHit->b2);
	//		else
	//			surfaceColor = Spectrum(1.f, 1.f, 1.f);
	// Interpolate face normal
	//      N = mesh->InterpolateTriNormal(triIndex, rayHit->b1, rayHit->b2);

	//########

	currentMeshIndex = ss->meshIDs[currentTriangleIndex];
	triIndex = currentTriangleIndex
			- ss->meshFirstTriangleOffset[currentMeshIndex];

	POINTERFREESCENE::Mesh m = ss->meshDescs[currentMeshIndex];

	//SSCENE::Material *hitPointMat = &ss->mats[ss->meshMats[currentMeshIndex]];

	//		if (mesh->HasColors())
	//		for (int i = 0; i < mesh->GetTotalVertexCount(); i++) {
	//			if (mesh->GetColor(i).r != ss->colors[m.colorsOffset + i].r ||
	//					mesh->GetColor(i).g != ss->colors[m.colorsOffset + i].g ||
	//					mesh->GetColor(i).b  != ss->colors[m.colorsOffset + i].b) {
	//				printf("asdasdad");
	//			}
	//		}

	if (m.hasColors) {

		ss->Mesh_InterpolateColor((Spectrum*) &ss->colors[m.colorsOffset],
				&ss->tris[m.trisOffset], triIndex, rayHit->b1, rayHit->b2,
				&surfaceColor);

	} else {
		surfaceColor = Spectrum(1.f, 1.f, 1.f);
	}

	ss->Mesh_InterpolateNormal(&ss->normals[m.vertsOffset],
			&ss->tris[m.trisOffset], triIndex, rayHit->b1, rayHit->b2, N);

	//		// Check if I have to apply texture mapping or normal mapping
	//		TexMapInstance *tm =
	//				scene->objectTexMaps[currentMeshIndex];
	//
	//
	//
	//		BumpMapInstance *bm =
	//				scene->objectBumpMaps[currentMeshIndex];
	//
	//		NormalMapInstance *nm =
	//				scene->objectNormalMaps[currentMeshIndex];
	//
	//		if (tm || bm || nm) {
	//			// Interpolate UV coordinates if required
	//
	//
	//			//const UV triUV = mesh->InterpolateTriUV(triIndex, rayHit->b1, rayHit->b2);
	//
	//
	//			UV triUV;
	//			ss->Mesh_InterpolateUV(&ss->uvs[m.vertsOffset],
	//					&ss->tris[m.trisOffset], triIndex, rayHit->b1,
	//					rayHit->b2, &triUV);
	//
	//			// Check if there is an assigned texture map
	//			if (tm) {
	//				const TextureMap *map = tm->GetTexMap();
	//
	//				// Apply texture mapping
	//				surfaceColor *= map->GetColor(triUV);
	//
	//				// Check if the texture map has an alpha channel
	//				if (map->HasAlpha()) {
	//					const float alpha = map->GetAlpha(triUV);
	//
	//					if ((alpha == 0.0f) || ((alpha < 1.f)
	//							&& (rndGen->floatValue() > alpha))) {
	//						*ray = Ray(hitPoint, ray->d);
	//						return true;
	//					}
	//				}
	//			}
	//
	//			// Check if there is an assigned bump/normal map
	//			if (bm || nm) {
	//				if (nm) {
	//					// Apply normal mapping
	//					const Spectrum color = nm->GetTexMap()->GetColor(
	//							triUV);
	//
	//					const float x = 2.0 * (color.r - 0.5);
	//					const float y = 2.0 * (color.g - 0.5);
	//					const float z = 2.0 * (color.b - 0.5);
	//
	//					Vector v1, v2;
	//					CoordinateSystem(Vector(N), &v1, &v2);
	//					N = Normalize(
	//							Normal(v1.x * x + v2.x * y + N.x * z,
	//									v1.y * x + v2.y * y + N.y * z,
	//									v1.z * x + v2.z * y + N.z * z));
	//				}
	//
	//				if (bm) {
	//					// Apply bump mapping
	//					const TextureMap *map = bm->GetTexMap();
	//					const UV &dudv = map->GetDuDv();
	//
	//					const float b0 = map->GetColor(triUV).Filter();
	//
	//					const UV uvdu(triUV.u + dudv.u, triUV.v);
	//					const float bu = map->GetColor(uvdu).Filter();
	//
	//					const UV uvdv(triUV.u, triUV.v + dudv.v);
	//					const float bv = map->GetColor(uvdv).Filter();
	//
	//					const float scale = bm->GetScale();
	//					const Vector bump(scale * (bu - b0),
	//							scale * (bv - b0), 1.f);
	//
	//					Vector v1, v2;
	//					CoordinateSystem(Vector(N), &v1, &v2);
	//					N = Normalize(
	//							Normal(
	//									v1.x * bump.x + v2.x * bump.y + N.x
	//											* bump.z,
	//									v1.y * bump.x + v2.y * bump.y + N.y
	//											* bump.z,
	//									v1.z * bump.x + v2.z * bump.y + N.z
	//											* bump.z));
	//				}
	//			}
	//		}

	// Flip the normal if required
	if (Dot(ray->d, N) > 0.f)
		shadeN = -N;
	else
		shadeN = N;

	return false;
}

void Engine::InitPhotonPath(PointerFreeScene* ss, PhotonPath *photonPath,
		Ray *ray, Seed& seed) {

	//Scene *scene = ss->scene;
	// Select one light source
	float lpdf;
	float pdf;

	Spectrum f;

	//photonPath->seed = mwc();

	//seed = mwc();

	float u0 = getFloatRNG(seed);
	float u1 = getFloatRNG(seed);
	float u2 = getFloatRNG(seed);
	float u3 = getFloatRNG(seed);
	float u4 = getFloatRNG(seed);
	float u5 = getFloatRNG(seed);

//		float u0 = getFloatRNG2(seed);
//		float u1 = getFloatRNG2(seed);
//		float u2 = getFloatRNG2(seed);
//		float u3 = getFloatRNG2(seed);
//		float u4 = getFloatRNG2(seed);
//		float u5 = getFloatRNG2(seed);

	int lightIndex;
	POINTERFREESCENE::LightSourceType lightT = ss->SampleAllLights(u0, &lpdf,
			lightIndex, ss->infiniteLight, ss->sunLight, ss->skyLight);

	if (lightT == POINTERFREESCENE::TYPE_IL_IS)
		ss->InfiniteLight_Sample_L(u1, u2, u3, u4, u5, &pdf, ray,
				photonPath->flux, ss->infiniteLight, ss->infiniteLightMap);

	else if (lightT == POINTERFREESCENE::TYPE_SUN)
		ss->SunLight_Sample_L(u1, u2, u3, u4, u5, &pdf, ray, photonPath->flux,
				ss->sunLight);

	else if (lightT == POINTERFREESCENE::TYPE_IL_SKY)
		ss->SkyLight_Sample_L(u1, u2, u3, u4, u5, &pdf, ray, photonPath->flux,
				ss->skyLight);

	else {
		ss->TriangleLight_Sample_L(&ss->areaLights[lightIndex], u1, u2, u3, u4,
				u5, &pdf, ray, photonPath->flux, &ss->colors[0],
				&ss->meshDescs[0]);
	}

	//##########

	//const LightSource *light2 = scene->SampleAllLights(u0, &lpdf);

	// Initialize the photon path
	//photonPath->flux = light2->Sample_L(scene, u1, u2, u3, u4, u5, &pdf, ray);

	//#########

	photonPath->flux /= pdf * lpdf;
	photonPath->depth = 0;

	//incPhotonCount();
}

void PPMEngine::InitRadius(uint iteration, Worker* w) {

	BBox* hitPointsbbox = w->GetHostBBox();

	Vector ssize = hitPointsbbox->pMax - hitPointsbbox->pMin;

	float photonRadius = ((ssize.x + ssize.y + ssize.z) / 3.f)
			/ ((cfg->width * cfg->superSampling
					+ cfg->height * cfg->superSampling) / 2.f) * 2.f;

	float photonRadius2 = photonRadius * photonRadius;

	w->SetNonPAInitialRadius2(photonRadius2);

}

void SPPMEngine::InitRadius(uint iteration, Worker* w) {

	BBox* hitPointsbbox = w->GetHostBBox();

	Vector ssize = hitPointsbbox->pMax - hitPointsbbox->pMin;

	float photonRadius = ((ssize.x + ssize.y + ssize.z) / 3.f)
			/ ((cfg->width * cfg->superSampling
					+ cfg->height * cfg->superSampling) / 2.f) * 2.f;

	float photonRadius2 = photonRadius * photonRadius;

	w->SetNonPAInitialRadius2(photonRadius2);

}

void PPMPAEngine::InitRadius(uint iteration, Worker* w) {

	BBox* hitPointsbbox = w->GetHostBBox();

	Vector ssize = hitPointsbbox->pMax - hitPointsbbox->pMin;

	float photonRadius = ((ssize.x + ssize.y + ssize.z) / 3.f)
			/ ((cfg->width * cfg->superSampling
					+ cfg->height * cfg->superSampling) / 2.f) * 2.f;

	float photonRadius2 = photonRadius * photonRadius;

	float g = 1;
	for (uint k = 1; k < iteration; k++)
		g *= (k + cfg->alpha) / k;

	g /= iteration;

	photonRadius2 = photonRadius2 * g;

	w->currentPhotonRadius2 = photonRadius2;

}

void SPPMPAEngine::InitRadius(uint iteration, Worker* w) {

	BBox* hitPointsbbox = w->GetHostBBox();

	Vector ssize = hitPointsbbox->pMax - hitPointsbbox->pMin;

	float photonRadius = ((ssize.x + ssize.y + ssize.z) / 3.f)
			/ ((cfg->width * cfg->superSampling
					+ cfg->height * cfg->superSampling) / 2.f) * 2.f;

	float photonRadius2 = photonRadius * photonRadius;

	float g = 1;
	for (uint k = 1; k < iteration; k++)
		g *= (k + cfg->alpha) / k;

	g /= iteration;

	photonRadius2 = photonRadius2 * g;

	w->currentPhotonRadius2 = photonRadius2;

}

void PPMEngine::ProcessIterations(Worker* worker, bool buildHitPoints) {

	u_int64_t photonPerIteration = cfg->photonsFirstIteration;

	uint iterationCount;

	__p.reg("Build Hit Points");
	if (buildHitPoints)
		worker->BuildHitPoints(1);
	__p.stp("Build Hit Points");

	__p.reg("Update BBox");
	worker->UpdateBBox();
	__p.stp("Update BBox");

	__p.reg("Initialize radius");
	InitRadius(0, worker);
	__p.stp("Initialize radius");

	while (!boost::this_thread::interruption_requested()) {

		double start = WallClockTime();

		__p.lsstt("Process Iterations > Iterations");

		iterationCount = IncIteration();

		__p.lsstp("Process Iterations > Iterations");

		if (iterationCount > MAX_ITERATIONS)
			break;

		__p.lsstt("Process Iterations > Iterations");

		photonPerIteration = cfg->photonsFirstIteration;

		fprintf(stderr, "Device %d: Processing iteration %d with %lu photons\n",
				worker->getDeviceID(), iterationCount, photonPerIteration);

		__p.lsstt("Process Iterations > Iterations > Update lookup");

		worker->UpdateQueryRangeLookupAcc(iterationCount);

		__p.lsstp("Process Iterations > Iterations > Update lookup");

		__p.lsstt("Process Iterations > Iterations > Build Photon Map");

		photonPerIteration = worker->BuildPhotonMap(photonPerIteration);

		__p.lsstp("Process Iterations > Iterations > Build Photon Map");

		__p.lsstt("Process Iterations > Iterations > Radiance calc");

		worker->AccumulateFluxPPM(iterationCount, photonPerIteration);

		__p.lsstp("Process Iterations > Iterations > Radiance calc");

		__p.lsstt("Process Iterations > Iterations > Update Samples");

		UpdateSampleFrameBuffer(photonPerIteration, worker);

		__p.lsstp("Process Iterations > Iterations > Update Samples");

		/**
		 * iteration lock required in PhotonTracedTotal
		 */
		incPhotonTracedTotal(photonPerIteration);

		worker->UpdateProfiler(iterationCount, start);

		__p.lsstp("Process Iterations > Iterations");

		if (iterationCount % 10 == 0)
			SaveImpl(to_string<uint>(iterationCount, std::dec) + cfg->fileName);

	}

	worker->profiler->printStats(worker->deviceID);

}

void SPPMEngine::ProcessIterations(Worker* worker, bool buildHitPoints) {

	u_int64_t photonPerIteration = cfg->photonsFirstIteration;

	uint iterationCount;

	bool firstIteration = true;

	while (!boost::this_thread::interruption_requested()) {

		double start = WallClockTime();

		__p.lsstt("Process Iterations > Iterations");

		iterationCount = IncIteration();

		__p.lsstp("Process Iterations > Iterations");

		if (iterationCount > MAX_ITERATIONS)
			break;

		__p.lsstt("Process Iterations > Iterations");

		photonPerIteration = cfg->photonsFirstIteration;

		fprintf(stderr, "Device %d: Processing iteration %d with %lu photons\n",
				worker->getDeviceID(), iterationCount, photonPerIteration);

		__p.lsstt("Process Iterations > Iterations > Build Hit Points");

		worker->BuildHitPoints(iterationCount);

		__p.lsstp("Process Iterations > Iterations > Build Hit Points");

		__p.lsstt("Process Iterations > Iterations > Update BBox");

		worker->UpdateBBox();

		__p.lsstp("Process Iterations > Iterations > Update BBox");

		__p.lsstt("Process Iterations > Iterations > Initialize radius");

		if (firstIteration){
			InitRadius(iterationCount, worker);
			firstIteration = false;
		}

		__p.lsstp("Process Iterations > Iterations > Initialize radius");

		__p.lsstt("Process Iterations > Iterations > Update lookup");

		worker->BuildLookupAcc();

		__p.lsstp("Process Iterations > Iterations > Update lookup");

		__p.lsstt("Process Iterations > Iterations > Build Photon Map");

		photonPerIteration = worker->BuildPhotonMap(photonPerIteration);

		__p.lsstp("Process Iterations > Iterations > Build Photon Map");

		__p.lsstt("Process Iterations > Iterations > Radiance calc");

		worker->AccumulateFluxSPPM(iterationCount, photonPerIteration);

		__p.lsstp("Process Iterations > Iterations > Radiance calc");

		__p.lsstt("Process Iterations > Iterations > Update Samples");

		UpdateSampleFrameBuffer(photonPerIteration, worker);

		__p.lsstp("Process Iterations > Iterations > Update Samples");

		/**
		 * iteration lock required in PhotonTracedTotal
		 */
		incPhotonTracedTotal(photonPerIteration);

		worker->UpdateProfiler(iterationCount, start);

		__p.lsstp("Process Iterations > Iterations");

		if (iterationCount % 50 == 0)
			SaveImpl(to_string<uint>(iterationCount, std::dec) + cfg->fileName);

	}

	worker->profiler->printStats(worker->deviceID);

}

void PPMPAEngine::ProcessIterations(Worker* worker, bool buildHitPoints) {

	u_int64_t photonPerIteration = cfg->photonsFirstIteration;

	uint iterationCount;

	__p.reg("Build Hit Points");

	if (buildHitPoints) {
		worker->BuildHitPoints(1);


		if (cfg->ndevices > 1) {
			__p.lsstt("Cpy HPs to device");
			worker->updateDeviceHitPointsInfo(buildHitPoints);
			__p.lsstp("Cpy HPs to device");
		}

	}
	__p.stp("Build Hit Points");

	waitForHitPoints->wait();

	if (!buildHitPoints) {
		__p.lsstt("Cpy HPs to device");
		worker->updateDeviceHitPointsInfo(0);
		__p.lsstp("Cpy HPs to device");
	}

	__p.reg("Update BBox");
	worker->UpdateBBox();
	__p.stp("Update BBox");

	while (!boost::this_thread::interruption_requested()) {

		double start = WallClockTime();

		__p.lsstt("Process Iterations > Iterations");
		iterationCount = IncIteration();
		__p.lsstp("Process Iterations > Iterations");

		if (iterationCount > MAX_ITERATIONS)
			break;

		__p.lsstt("Process Iterations > Iterations");

		photonPerIteration = cfg->photonsFirstIteration;

		__p.lsstt("Process Iterations > Initialize radius");
		InitRadius(iterationCount, worker);
		__p.lsstp("Process Iterations > Initialize radius");

		fprintf(stderr,
				"Device %d: Processing iteration %d with %lu photons, radius: %.8f\n",
				worker->getDeviceID(), iterationCount, photonPerIteration,
				worker->currentPhotonRadius2);

		__p.lsstt("Process Iterations > Iterations > Update lookup");
		worker->UpdateQueryRangeLookupAcc(iterationCount);
		__p.lsstp("Process Iterations > Iterations > Update lookup");

		__p.lsstt("Process Iterations > Iterations > Build Photon Map");
		photonPerIteration = worker->BuildPhotonMap(photonPerIteration);
		__p.lsstp("Process Iterations > Iterations > Build Photon Map");

		__p.lsstt("Process Iterations > Iterations > Radiance calc");
		worker->AccumulateFluxPPMPA(iterationCount, photonPerIteration);
		__p.lsstp("Process Iterations > Iterations > Radiance calc");

		__p.lsstt("Process Iterations > Iterations > Update Samples");
		UpdateSampleFrameBuffer(photonPerIteration, worker);
		__p.lsstp("Process Iterations > Iterations > Update Samples");

		/**
		 * iteration lock required in PhotonTracedTotal
		 */
		incPhotonTracedTotal(photonPerIteration);

		worker->UpdateProfiler(iterationCount, start);

		__p.lsstp("Process Iterations > Iterations");

		if (iterationCount % 10 == 0)
			SaveImpl(to_string<uint>(iterationCount, std::dec) + cfg->fileName);

	}

	worker->profiler->printStats(worker->getDeviceID());

}

void SPPMPAEngine::ProcessIterations(Worker* worker, bool buildHitPoints) {

	u_int64_t photonPerIteration = cfg->photonsFirstIteration;

	uint iterationCount;

	while (!boost::this_thread::interruption_requested()) {

		double start = WallClockTime();

		__p.lsstt("Process Iterations > Iterations");

		iterationCount = IncIteration();

		__p.lsstp("Process Iterations > Iterations");

		if (iterationCount > MAX_ITERATIONS)
			break;

		__p.lsstt("Process Iterations > Iterations");

		photonPerIteration = cfg->photonsFirstIteration;

		//fprintf(stderr, "Device %d: Processing iteration %d with %lu photons\n",
		//		worker->getDeviceID(), iterationCount, photonPerIteration);

		__p.lsstt("Process Iterations > Iterations > Build Hit Points");
		worker->BuildHitPoints(iterationCount);
		__p.lsstp("Process Iterations > Iterations > Build Hit Points");

		__p.lsstt("Process Iterations > Iterations > Update BBox");
		worker->UpdateBBox();
		__p.lsstp("Process Iterations > Iterations > Update BBox");

		__p.lsstt("Process Iterations > Iterations > Initialize radius");
		InitRadius(iterationCount, worker);
		__p.lsstp("Process Iterations > Iterations > Initialize radius");

		fprintf(stderr, "Device %d: Iteration %d radius: %.8f\n",
				worker->getDeviceID(), iterationCount,
				worker->currentPhotonRadius2);

		__p.lsstt("Process Iterations > Iterations > Update lookup");
		worker->BuildLookupAcc();
		__p.lsstp("Process Iterations > Iterations > Update lookup");

		__p.lsstt("Process Iterations > Iterations > Build Photon Map");
		photonPerIteration = worker->BuildPhotonMap(photonPerIteration);
		__p.lsstp("Process Iterations > Iterations > Build Photon Map");

		__p.lsstt("Process Iterations > Iterations > Radiance calc");
		worker->AccumulateFluxSPPMPA(iterationCount, photonPerIteration);
		__p.lsstp("Process Iterations > Iterations > Radiance calc");

		__p.lsstt("Process Iterations > Iterations > Update Samples");
		UpdateSampleFrameBuffer(photonPerIteration, worker);
		__p.lsstp("Process Iterations > Iterations > Update Samples");

		worker->ResetDeviceHitPointsInfo();

		/**
		 * iteration lock required in PhotonTracedTotal
		 */
		incPhotonTracedTotal(photonPerIteration);

		worker->UpdateProfiler(iterationCount, start);

		__p.lsstp("Process Iterations > Iterations");

		if (iterationCount % 50 == 0)
			SaveImpl(to_string<uint>(iterationCount, std::dec) + cfg->fileName);

	}

	worker->profiler->printStats(worker->getDeviceID());

}
