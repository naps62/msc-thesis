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
#include <boost/thread.hpp>
#include "cppbench.h"

using namespace boost;

CPPBENCH __p;
RenderConfig* cfg;

static void Draw(int argc, char *argv[]) {
	InitGlut(argc, argv, cfg->width, cfg->height);

	RunGlut(cfg->width, cfg->height);

}

inline void createWorker(devicesType d, Engine* engine) {

	Worker* w;

	switch (d) {
	case CPU0:
		w = new CPU_Worker(CPU0, engine);
		cfg->devices->insert(std::pair<devicesType, Worker*>(CPU0, w));
		cfg->ndevices++;
		break;
	case GPU0:
		w = new CUDA_Worker(GPU0, engine);
		cfg->devices->insert(std::pair<devicesType, Worker*>(GPU0, w));
		cfg->ndevices++;
		break;
	case GPU2:
		w = new CUDA_Worker(GPU2, engine);
		cfg->devices->insert(std::pair<devicesType, Worker*>(GPU2, w));
		cfg->ndevices++;
		break;
	default:
		printf("Device initizalizaiton code not specified\n");
		assert(false);
		break;
	}

}

inline void createWorkers() {

	Engine* engine = cfg->GetEngine();
	uint dev_config = cfg->device_configuration;

	switch (dev_config) {
	case 0:
		createWorker(CPU0, engine);
		fprintf(stderr, "Using CPU0\n");
		break;
	case 1:
		createWorker(GPU0, engine);
		fprintf(stderr, "Using GPU0\n");
		break;
	case 2:
		createWorker(CPU0, engine);
		createWorker(GPU0, engine);
		fprintf(stderr, "Using CPU0 + GPU0\n");
		break;
	case 3:
		createWorker(GPU0, engine);
		createWorker(GPU2, engine);
		fprintf(stderr, "Using GPU0 + GPU2\n");
		break;
	case 4:
		createWorker(CPU0, engine);
		createWorker(GPU0, engine);
		createWorker(GPU2, engine);
		fprintf(stderr, "Using CPU0 + GPU0 + GPU2\n");
		break;

	default:
		break;
	}
}

int main(int argc, char *argv[]) {

	__p.reg("Walltime");

	srand(1000);

//	printf("siz %lu\n", sizeof(HitPointPositionInfo));
//	printf("siz %lu\n", sizeof(HitPointRadianceFlux));
//		printf("siz %lu\n", sizeof(PhotonHit));

	cfg = new RenderConfig();

	/**************************************************/
	cfg->width = 640;
	cfg->height = 480;
	cfg->superSampling = sqrt(9);
	cfg->photonsFirstIteration =(1 << 21);
	//cfg->photonsFirstIteration = (1 << 21) + (1 << 21); //1.5M;
	//cfg->photonsFirstIteration =2048*10; //1.5M;
	cfg->alpha = 0.7f;

	//cfg->enginetype = PPM;
	//cfg->enginetype = SPPM;
	//cfg->enginetype = PPMPA;
	cfg->enginetype = SPPMPA;

	cfg->device_configuration = 0; //CPU
	//cfg->device_configuration = 1; //GPU
	//cfg->device_configuration = 2; //CPU+GPU
	//cfg->device_configuration = 3; //GPU+GPU
	//cfg->device_configuration = 4; //CPU+GPU+GPU

	cfg->rebuildHash = 0;

	cfg->fileName = "image.png";

	 std::string sceneFileName = "scenes/kitchen/kitchen.scn";
	//	std::string sceneFileName = "scenes/alloy/alloy.scn";
	//	std::string sceneFileName = "scenes/bigmonkey/bigmonkey.scn";
	 // std::string sceneFileName = "scenes/psor-cube/psor-cube.scn";
	//	std::string sceneFileName = "scenes/classroom/classroom.scn";
	//	std::string sceneFileName = "scenes/luxball/luxball.scn";
	//	std::string sceneFileName = "scenes/cornell/cornell.scn";
	//	std::string sceneFileName = "scenes/simple/simple.scn";
	//	std::string sceneFileName = "scenes/simple-mat/simple-mat.scn";
	//	std::string sceneFileName = "scenes/sky/sky.scn";
	//	std::string sceneFileName = "scenes/studiotest/studiotest.scn";

	/**************************************************/

	if (cfg->GetEngineType() == SPPM || cfg->GetEngineType() == SPPMPA)
		cfg->superSampling = 1;
	//cfg->photonsFirstIteration = (1 << 20);

	cfg->hitPointTotal = cfg->width * cfg->height * cfg->superSampling
			* cfg->superSampling;

	Engine* engine;
	switch (cfg->enginetype) {
	case PPM:
		engine = new PPMEngine();
		fprintf(stderr, "Using PPM engine\n");
		break;
	case SPPM:
		engine = new SPPMEngine();
		fprintf(stderr, "Using SPPM engine\n");
		break;
	case PPMPA:
		engine = new PPMPAEngine();
		fprintf(stderr, "Using PPMPA engine\n");
		break;
	case SPPMPA:
		engine = new SPPMPAEngine();
		fprintf(stderr, "Using SPPMPA engine\n");
		break;
	default:
		break;
	}

	cfg->engine = engine;

	engine->ss = new PointerFreeScene(cfg->width, cfg->height, sceneFileName);

	createWorkers();

	if (cfg->GetEngineType() == PPM || cfg->GetEngineType() == SPPM)
		assert(cfg->ndevices == 1);

#ifdef ENABLE_TIME_BREAKDOWN
	assert(cfg->ndevices == 1);
#endif

	//Only one device builds the hitpoints
	if (cfg->GetEngineType() == PPMPA)
		((PPMPAEngine*) engine)->waitForHitPoints = new boost::barrier(
				cfg->ndevices);

	__p.reg("Total Job");

	cfg->startTime = WallClockTime();

	bool build_hit = true;
	std::map<devicesType, Worker*>::iterator devs;
	for (devs = cfg->devices->begin(); devs != cfg->devices->end(); devs++) {
		devs->second->Start(build_hit);
		build_hit = false;
	}

#ifdef USE_GLUT
	engine->draw_thread = new boost::thread(Draw, argc, argv);
#endif

	for (devs = cfg->devices->begin(); devs != cfg->devices->end(); devs++) {
		devs->second->thread->join();
	}

	__p.stp("Total Job");
	const double elapsedTime = WallClockTime() - cfg->startTime;

	float MPhotonsSec = engine->getPhotonTracedTotal()
			/ (elapsedTime * 1000000.f);

	const float itsec = engine->GetIterationNumber() / elapsedTime;

	printf("System excution time|%.3f\n", elapsedTime);
	printf("System MPhotons/sec|%.3f\n", MPhotonsSec);
	printf("System iteration/sec|%.3f\n", itsec);
	printf("Total photons: %.2fM\n",
			engine->getPhotonTracedTotal() / 1000000.f);

	//printf("Splat flux call count: %d\n", cpuWorker->lookupA->call_times);

	__p.stp("Walltime");

#ifdef ENABLE_TIME_BREAKDOWN
	__p.PRINTALL_SECONDS();
#endif

#ifdef USE_GLUT
	engine->draw_thread->join();
#endif

	return EXIT_SUCCESS;
}
