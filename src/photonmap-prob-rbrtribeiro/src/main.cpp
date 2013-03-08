/*
 * job.cpp
 *
 *  Created on: Jul 25, 2012
 *      Author: rr
 */

#include "renderEngine.h"

#include <GL/glut.h>
#include <FreeImage.h>
#include <boost/detail/container_fwd.hpp>
#include "luxrays/utils/sdl/scene.h"
#include "CUDA_Worker.h"
#include "CPU_Worker.h"
#include "config.h"

//#include "cppbench.h"

//CPPBENCH __BENCH;

static void Draw(int argc, char *argv[]) {
	InitGlut(argc, argv, engine->width, engine->height);

	RunGlut(engine->width, engine->height);

}

int main(int argc, char *argv[]) {

	srand(1000);

	float alpha = 0.7f;
	uint width;
	uint height;
	uint superSampling;
	unsigned long long photonsFirstIteration;

#ifdef RENDER_FAST_PHOTON
	alpha = alpha;
	width = 640;
	height = 480;
	superSampling = 4;
	photonsFirstIteration = 1 << 19;//0.5M;
#endif

#ifdef RENDER_TINY
	alpha = alpha;
	width = 480;
	height = 480;
	superSampling = 4;
	photonsFirstIteration = 1 << 21;//0.5M;
#endif

#ifdef RENDER_MEDIUM
	alpha = alpha;
	width = 640;
	height = 480;
	superSampling = 3;
	photonsFirstIteration = 1 << 21;//2M
#endif

#ifdef RENDER_BIG
	alpha = alpha;
	width = 640;
	height = 480;
	superSampling = 4;
	photonsFirstIteration = 1 << 22;//4M
#endif

#ifdef RENDER_HUGE
	alpha = alpha;
	width = 640;
	height = 480;
	superSampling = 6;
	photonsFirstIteration = 1 << 23;//8M
#endif

#if defined USE_SPPM || defined USE_SPPMPA
	superSampling=1;
#endif

//	__BENCH.REGISTER("Total Job");

	size_t hitPointTotal = width * height * superSampling * superSampling;

	uint ndvices = 0;

#ifdef GPU0
	ndvices++;
#endif
#ifdef GPU2
	ndvices++;
#endif
#ifdef CPU
	ndvices++;
#endif

#if defined USE_PPM || defined USE_SPPM
	if (ndvices > 1) assert(false);
#endif

#ifndef DISABLE_TIME_BREAKDOWN
	if (ndvices > 1) assert(false);
#endif

	engine = new PPM(alpha, width, height, superSampling, photonsFirstIteration, ndvices);

	std::string sceneFileName = "scenes/kitchen/kitchen.scn";

	engine->fileName = "kitchen.png";

	//	std::string sceneFileName = "scenes/alloy/alloy.scn";
	//	std::string sceneFileName = "scenes/bigmonkey/bigmonkey.scn";
	//  std::string sceneFileName = "scenes/psor-cube/psor-cube.scn";
	//	std::string sceneFileName = "scenes/classroom/classroom.scn";
	//	std::string sceneFileName = "scenes/luxball/luxball.scn";nao
	//	std::string sceneFileName = "scenes/cornell/cornell.scn";
	//	std::string sceneFileName = "scenes/simple/simple.scn";
	//	std::string sceneFileName = "scenes/simple-mat/simple-mat.scn";
	//	std::string sceneFileName = "scenes/sky/sky.scn";
	//	std::string sceneFileName = "scenes/studiotest/studiotest.scn";

	engine->ss = new PointerFreeScene(width, height, sceneFileName);

	engine->startTime = WallClockTime();

	Seed* seedBuffer;
	uint devID;
	uint c;
	bool build_hit = false;

#ifdef GPU0
	devID = 0;
	size_t WORK_BUCKET_SIZE_GPU0 = SM * FACTOR * BLOCKSIZE; // SMs*FACTOR*THEADSBLOCK
	c = max(hitPointTotal, WORK_BUCKET_SIZE_GPU0);
	seedBuffer = new Seed[c];
	for (uint i = 0; i < c; i++)
		seedBuffer[i] = mwc(i+devID);
	build_hit = true;
	CUDA_Worker* gpuWorker0 = new CUDA_Worker(0, engine->ss, WORK_BUCKET_SIZE_GPU0, seedBuffer,
			build_hit);
#endif

#ifdef GPU2
	devID = 2;
	size_t WORK_BUCKET_SIZE_GPU2 = SM * FACTOR * BLOCKSIZE; // SMs*FACTOR*THEADSBLOCK
	c = max(hitPointTotal, WORK_BUCKET_SIZE_GPU2);
	seedBuffer = new Seed[c];
	for (uint i = 0; i < c; i++)
		seedBuffer[i] = mwc(i+devID);
	if (build_hit)
		build_hit = false;
	else
		build_hit = true;
	CUDA_Worker* gpuWorker2 = new CUDA_Worker(2, engine->ss, WORK_BUCKET_SIZE_GPU2, seedBuffer,
			build_hit);
	build_hit = true;
#endif

#ifdef CPU
	devID = 100;

	size_t WORK_BUCKET_SIZE_CPU = 1024 * 256;
	c = max(hitPointTotal, WORK_BUCKET_SIZE_CPU);
	seedBuffer = new Seed[c];

	for (uint i = 0; i < c; i++)
		seedBuffer[i] = mwc(i+devID);

	if (build_hit)
		build_hit = false;
	else
		build_hit = true;

	CPU_Worker* cpuWorker = new CPU_Worker(devID, engine->ss, WORK_BUCKET_SIZE_CPU, seedBuffer,
			build_hit);
#endif

#ifdef USE_GLUT
	engine->draw_thread = new boost::thread(boost::bind(Draw, argc, argv));
#endif

#ifdef GPU0
	gpuWorker0->thread->join();
#endif
#ifdef GPU2
	gpuWorker2->thread->join();
#endif
#ifdef CPU
	cpuWorker->thread->join();

#endif

	const double elapsedTime = WallClockTime() - engine->startTime;
	float MPhotonsSec = engine->getPhotonTracedTotal() / (elapsedTime * 1000000.f);
	const float itsec = engine->GetIterationNumber() / elapsedTime;

	printf("Avg. %.2f MPhotons/sec\n", MPhotonsSec);
	printf("Avg. %.3f iteration/sec\n", itsec);
	printf("Total photons: %.2fM\n", engine->getPhotonTracedTotal() / 1000000.f);

//	__BENCH.STOP("Total Job");
//	__BENCH.PRINTALL_SECONDS();

#ifdef USE_GLUT
	engine->draw_thread->join();
#endif


	return EXIT_SUCCESS;
}
