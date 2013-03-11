/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

#include <gama.h>
#include <cstdlib>

#include <cublas.h>

#include "saxpy.h"

MemorySystem* LowLevelMemAllocator::_memSys = NULL;

#define N 100000


int main() {
	RuntimeScheduler* rs = new RuntimeScheduler();

	// allocation
	RXY rxy(N);
	float alpha = 1.5f;

	// initialize with random data
	srand(7);
	for(int i = 0; i < N; ++i) {
		rxy.r[i] = 0.f;
		rxy.x[i] = 1.5f;
		rxy.y[i] = float(rand()) / RAND_MAX + 1;
	}

	for(int i = 0; i < N; ++i) {
		rxy.r[i] = 0.f;
	}

	saxpy* s = new saxpy(rxy, alpha, N, 0, N);
	rs->synchronize();
	double start = getTimeMS();
	rs->submit(s);
	rs->synchronize();
	double end = getTimeMS();

	unsigned int count = 0;

	for(int i = 0; i < N; ++i) {
		if (fabs(rxy.r[i] - (rxy.x[i] * alpha + rxy.y[i])) > FLT_EPSILON*4) {
			count++;
		}
	}

	double start_m = getTimeMS();
	cublasSaxpy(N, alpha, rxy.x.getPtr(), 1, rxy.y.getPtr(), 1);
	cudaDeviceSynchronize();
	double end_m = getTimeMS();
	printf("Time blas Kernel: %.4f\n", (end_m - start_m));
	printf("Time GAMA: %.4f, Error: %d (%d) -- %.3f\% \n", (end - start), count, N, (float(count)/float(N)) *100);
	delete rs;
}
