/*
 * job.cpp
 *
 *  Created on: Jul 25, 2012
 *      Author: rr
 */

#include "renderEngine.h"

#include <GL/glut.h>
#include <FreeImage.h>
#include <boost/thread.hpp>
#include <boost/detail/container_fwd.hpp>
#include "luxrays/utils/sdl/scene.h"
#include "CPU_Worker.h"
#include "config.h"

static void Draw(int argc, char *argv[]) {
	InitGlut(argc, argv, engine->width, engine->height);

	RunGlut(engine->width, engine->height);
}

const Config* config;

int main(int argc, char *argv[]) {

	// load configurations
	config = new Config("Options", argc, argv);

	srand(1000);

	float alpha = 0.7f;
	uint width;
	uint height;
	uint superSampling;
	unsigned long long photonsFirstIteration;

	//alpha = alpha;
	width = config->width;//640;
	height = config->height;//480;
	superSampling = config->spp;//1;
	photonsFirstIteration = 1 << config->photons_first_iter_exp;//0.5M;




#if defined USE_SPPM || defined USE_SPPMPA
	superSampling=1;
#endif


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

	engine = new PPM(alpha, width, height, superSampling, photonsFirstIteration, ndvices);

	std::string sceneFileName = config->scene_file.c_str(); //"scenes/kitchen/kitchen.scn";

	engine->fileName = config->output_file;//"kitchen.png";

	engine->ss = new PointerFreeScene(width, height, sceneFileName);

	engine->startTime = WallClockTime();
	Seed* seedBuffer;
	uint devID;
	uint c;
	bool build_hit = false;

	devID = 100;

	size_t WORK_BUCKET_SIZE_CPU = 1024 * 256;
	c = max((int)hitPointTotal, (int)WORK_BUCKET_SIZE_CPU);
	seedBuffer = new Seed[c];

	for (uint i = 0; i < c; i++)
		seedBuffer[i] = mwc(i+devID);

	if (build_hit)
		build_hit = false;
	else
		build_hit = true;

	CPU_Worker* cpuWorker = new CPU_Worker(devID, engine->ss, WORK_BUCKET_SIZE_CPU, seedBuffer,
			build_hit);

	if (config->use_display)
		engine->draw_thread = new boost::thread(boost::bind(Draw, argc, argv));


	cpuWorker->thread->join();

	const double elapsedTime = WallClockTime() - engine->startTime;
//	float MPhotonsSec = engine->getPhotonTracedTotal() / (elapsedTime * 1000000.f);
	const float itsec = engine->GetIterationNumber() / elapsedTime;

	if (config->use_display) {
		cout << "Done. waiting for display" << endl;
		engine->draw_thread->join();
	}

	engine->SaveImpl(config->output_file.c_str());

	fprintf(stderr, "Avg. %.3f iteration/sec\n", itsec);
	fprintf(stderr, "Total photons: %.2fM\n", engine->getPhotonTracedTotal() / 1000000.f);
	fprintf(stderr, "Total time:\n%f\n", elapsedTime);
	fflush(stdout);
	fflush(stderr);

	return EXIT_SUCCESS;
}
