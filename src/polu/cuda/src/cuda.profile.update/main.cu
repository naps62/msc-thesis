#include <iostream>
using namespace std;
#include <tk/stopwatch.hpp>
#include <cuda_runtime.h>

namespace profile {
	tk::Stopwatch *s;
	//PROFILE_COUNTER_CLASS * PROFILE_COUNTER_NAME;

	#define COUNT 7
	float up[COUNT];
	int count[COUNT];

	cudaEvent_t start_t, stop_t;

	void init() {
		s = new tk::Stopwatch();

		for(unsigned int i = 0; i < COUNT; ++i)
			up[i] = count[i] = 0;

		cudaEventCreate(&start_t);
		cudaEventCreate(&stop_t);
	}

	inline void output(std::ostream& out) {
		for(unsigned int i = 0; i < COUNT; ++i) {
			out << ((double)up[i]/(double)count[i]);
			if (i != COUNT - 1)
				out << ';';
		}
		out << endl;
	}

	void cleanup() {
		delete s;

		cudaEventDestroy(start_t);
		cudaEventDestroy(stop_t);
	}

	inline void time_up(int x) {
		//up[x] += s->last().microseconds();
		float elapsed;
		cudaEventElapsedTime(&elapsed, start_t, stop_t);
		up[x] += elapsed;
		count[x]++;
	}

	inline void cuda_start() {
		cudaEventRecord(start_t);
	}

	inline void cuda_stop() {
		cudaEventRecord(stop_t);
		cudaEventSynchronize(stop_t);
	}
}

#define NUM_ITERATIONS  200

#define PROFILE_COUNTER              profile::s
#define PROFILE_INIT()               profile::init()
#define PROFILE_OUTPUT()             profile::output(cout)
#define PROFILE_CLEANUP()            profile::cleanup()
#define PROFILE_START() profile::cuda_start()
#define PROFILE_STOP()  profile::cuda_stop()

#define PROFILE_RETRIEVE_UP(x) profile::time_up(x)


#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
#include "FVL/FVXMLReader.h"
#include "FVL/FVXMLWriter.h"
#include "FVio.h"
#include "FVL/FVParameters.h"
using namespace std;

#define  _CUDA 1
#include <cuda.h>
#include "kernels_cuda.cuh"

#define BLOCK_SIZE_UPDATE			512
#define GRID_SIZE(elems, threads)	((int) std::ceil((double)elems/threads))

#define _CUDA_ONLY      if (_CUDA)
#define _NO_CUDA_ONLY   if (!_CUDA)

/**
 * Parameters struct passed via xml file
 */
struct Parameters {
	string mesh_file;
	string velocity_file;
	string initial_file;
	string output_file;
	double final_time;
	double anim_time;
	int anim_jump;
	double dirichlet;
	double CFL;

	public:
	// Constructor receives parameter file
	Parameters(string parameters_filename) {
		FVL::FVParameters para(parameters_filename);

		this->mesh_file		= para.getString("MeshName");
		this->velocity_file	= para.getString("VelocityFile");
		this->initial_file	= para.getString("PoluInitFile");
		this->output_file	= para.getString("OutputFile");
		this->final_time	= para.getDouble("FinalTime");
		this->anim_time		= para.getDouble("AnimTimeStep");
		this->anim_jump		= para.getInteger("NbJump");
		this->dirichlet		= para.getDouble("DirichletCondition");
		this->CFL			= para.getDouble("CFL");
	}
};

int main(int argc, char **argv) {
	

	// var declaration
	int i = 0;
	double h, t, dt, v_max = 0;
	string name;

	// read params
	string param_filename;
	if (argc != 2) {
		param_filename = "param.xml";
	} else
		param_filename = argv[1];

	Parameters data(param_filename);

	// read mesh
	FVL::CFVMesh2D           mesh(data.mesh_file);
	FVL::CFVArray<double>    polution(mesh.num_cells);		// polution arrays
	FVL::CFVArray<double>    flux(mesh.num_edges);			// flux array
	FVL::CFVPoints2D<double> velocities(mesh.num_cells);	// velocities by cell (to calc vs array)
	FVL::CFVArray<double>    vs(mesh.num_edges);			// velocities by edge
	FVL::CFVMat<double> dummy(MAX_EDGES_PER_CELL, 1, mesh.num_cells);

	// read other input files
	FVL::FVXMLReader velocity_reader(data.velocity_file);
	FVL::FVXMLReader polu_ini_reader(data.initial_file);
	velocity_reader.getPoints2D(velocities, t, name);
	polu_ini_reader.getVec(polution, t, name);
	polu_ini_reader.close();
	velocity_reader.close();

	// compute velocity vector
	// TODO: Convert to CUDA
	#ifdef _CUDA
		kernel_compute_edge_velocities(mesh, velocities, vs, v_max);
		h = kernel_compute_mesh_parameter(mesh);
	#else
		cpu_compute_edge_velocities(mesh, velocities, vs, v_max);
		h = cpu_compute_mesh_parameter(mesh);
	#endif

	dt	= 1.0 / v_max * h;

	#ifdef _CUDA
		// saves whole mesh to CUDA memory
		mesh.cuda_malloc();
		polution.cuda_malloc();
		flux.cuda_malloc();
		vs.cuda_malloc();
		dummy.cuda_malloc();

		// data copy
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		mesh.cuda_save(stream);
		polution.cuda_save(stream);
		vs.cuda_save(stream);
	
		// block and grid sizes for each kernel
		dim3 grid_update(GRID_SIZE(mesh.num_cells, 512), 1, 1);
		dim3 block_update(512, 1, 1);

		dim3 grid_update2(GRID_SIZE(mesh.num_cells, 768), 1, 1);
		dim3 block_update2(768, 1, 1);
	#endif

	PROFILE_INIT();
	//
	// main loop start
	//
	for(unsigned int i = 0; i < NUM_ITERATIONS; ++i) {		
		PROFILE_START();
		kernel_update1<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt);
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP(0);

		PROFILE_START();
		kernel_update2<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt);
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP(1);

		PROFILE_START();
		kernel_update3<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt);
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP(2);

		PROFILE_START();
		kernel_update4<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt);
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP(3);

		PROFILE_START();
		kernel_update5<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt, dummy.cuda_get());
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP(4);

		PROFILE_START();
		kernel_update6<<< grid_update, block_update >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt, dummy.cuda_get());
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP(5);

		// same kernel, different block size
		PROFILE_START();
		kernel_update6<<< grid_update2, block_update2 >>>(mesh.cuda_get(), polution.cuda_get(), flux.cuda_get(), dt, dummy.cuda_get());
		PROFILE_STOP();
		PROFILE_RETRIEVE_UP(6);
	}

	#ifdef _CUDA
		vs.cuda_free();
		polution.cuda_free();
		flux.cuda_free();

		_DEBUG cudaCheckError(string("final check"));
	#endif

	// PROFILE ZONE --- measure postprocessing time
	
	PROFILE_STOP();
	PROFILE_OUTPUT();
	PROFILE_CLEANUP();
}

