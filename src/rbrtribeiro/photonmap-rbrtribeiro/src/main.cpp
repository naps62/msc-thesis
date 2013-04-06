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
#include "CUDA_Worker.h"
#include "CPU_Worker.h"
#include "config.h"

static void Draw(int argc, char *argv[]) {
	InitGlut(argc, argv, engine->width, engine->height);

	RunGlut(engine->width, engine->height);

}

//uint _num_threads = 0;
//uint _num_iters = 1;
//char* _render_cfg;
//char* _img_file;
const Config* config;

int main(int argc, char *argv[]) {

	// load configurations
	config = new Config("Options", argc, argv);

//	if (argc < 5) {
//		fprintf(stderr, "usage: program <num_threads> <num_iters> <render_cfg_file> <img_output_file>");
//		exit(0);
//	}
//	_num_threads = atoi(argv[1]);
//	_num_iters = atoi(argv[2]);
//	_render_cfg = argv[3];
//	_img_file = argv[4];

//	printf("num_threads: %d\n", _num_threads);
//	printf("num_iters: %d\n", _num_iters);
//	printf("render_cfg: %s\n", _render_cfg);
//	printf("img_file: %s\n", _img_file);


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

	engine->fileName = config->img_file;//"kitchen.png";

	//	std::string sceneFileName = "scenes/alloy/alloy.scn";
	//	std::string sceneFileName = "scenes/bigmonkey/bigmonkey.scn";
	//  std::string sceneFileName = "scenes/psor-cube/psor-cube.scn";
	//	std::string sceneFileName = "scenes/classroom/classroom.scn";
	//	std::string sceneFileName = "scenes/luxball/luxball.scn";
	//	std::string sceneFileName = "scenes/cornell/cornell.scn";
	//std::string sceneFileName = "scenes/simple/simple.scn";
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
#endif

	if (config->use_display)
		engine->draw_thread = new boost::thread(boost::bind(Draw, argc, argv));


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

	printf("Avg. %.3f iteration/sec\n", itsec);
	printf("Total photons: %.2fM\n", engine->getPhotonTracedTotal() / 1000000.f);

	if (config->use_display)
		engine->draw_thread->join();

	engine->SaveImpl(config->img_file.c_str());

	return EXIT_SUCCESS;
}
