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

#define N 10

//TODO add main function
int main() {
	RuntimeScheduler* rs = new RuntimeScheduler();

	// allocation
	smartPtr<float> R(sizeof(float) * N);
	smartPtr<float> X(sizeof(float) * N);
	smartPtr<float> Y(sizeof(float) * N);
	float alpha = 1.5f;

	// initialize with random data
	srand(7);
	for(int i = 0; i < N; ++i) {
		R[i] = 0.f;
		X[i] = 1.5f;
		Y[i] = float(rand()) / RAND_MAX + 1;
	}

	for(int i = 0; i < N; ++i) {
		R[i] = 0.f;
	}

	saxpy* s = new saxpy(R, X, Y, alpha, N, 0, N);
	rs->synchronize();
	double start = getTimeMS();
	rs->submit(s);
	rs->synchronize();
	double end = getTimeMS();

	unsigned int count = 0;

	for(int i = 0; i < N; ++i) {
		if (fabs(R[i] - (X[i] * alpha + Y[i])) > FLT_EPSILON*4) {
			count++;
		}
	}

	double start_m = getTimeMS();
	cublasSaxpy(N, alpha, X.getPtr(), 1, Y.getPtr(), 1);
	cudaDeviceSynchronize();
	double end_m = getTimeMS();
	printf("Time blas Kernel: %.4f\n", (end_m - start_m));
	printf("Time GAMA: %.4f, Error: %d (%d) -- %.3f\% \n", (end - start), count, N, (float(count)/float(N)) *100);
	delete rs;
}
