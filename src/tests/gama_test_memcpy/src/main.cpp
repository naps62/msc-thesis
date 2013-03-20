/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

#include <gama.h>
#include <cstdlib>

#include <cublas.h>

#include "kernel.h"

MemorySystem* LowLevelMemAllocator::_memSys = NULL;

#define N 100000

//TODO add main function
int main() {
	RuntimeScheduler* rs = new RuntimeScheduler();

	// allocation
	int cpu_arr[N];
	smartPtr<int> arr(sizeof(int) * N);
	float alpha = 1.5f;

	// initialize with random data
	for(int i = 0; i < N; ++i) {
		cpu_arr[i] = i;
	}

	kernel* s = new kernel(arr, N, 0, N);
	memcpy(arr.getPtr(), cpu_arr, sizeof(int) * N);
	rs->synchronize();
	rs->submit(s);
	rs->synchronize();

	unsigned int errors = 0;

	for(int i = 0; i < N; ++i) {
		if (arr[i] != i + 1) {
			std::cout << "arr[" << i << "] expected " << i + 1 << ", got " << arr[i] << std::endl;
			errors++;
		}
	}

	std::cout << errors << " values were wrong" << std::endl;

	delete rs;
}
