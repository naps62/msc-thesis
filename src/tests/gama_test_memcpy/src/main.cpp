/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

#include <gamalib/gamalib.h>
#include <cstdlib>

#include <cublas.h>

#include "kernel.h"
//#include "gama_ext/vector.h"

MemorySystem* LowLevelMemAllocator::_memSys = NULL;

#define N 100000

//TODO add main function
int main() {
	RuntimeScheduler* rs = new RuntimeScheduler();

	// allocation
	int cpu_arr[N];
	smartPtr<int> arr(N*sizeof(int));
	float alpha = 1.5f;

	// initialize with random data
	for(int i = 0; i < N; ++i) {
		cpu_arr[i] = i;
	}

	memcpy(arr.getPtr(), cpu_arr, sizeof(int) * N);
	rs->synchronize();
	rs->submit(new kernel(arr, N, 0, N));
	rs->synchronize();

	unsigned int errors = 0;

	for(int i = 0; i < N; ++i) {
		if (arr[i] != i + 1) {
			std::cout << "arr[" << i << "] expected " << i + 1 << ", got " << arr[i] << std::endl;
			errors++;
		}
	}
	std::cout << errors << " values were wrong" << std::endl;


	// initialize with random data
	for(int i = 0; i < N; ++i) {
		arr[i] = 0;
	}

	rs->submit(new kernel(arr, N, 0, N));
	rs->synchronize();

	std::cout << "second round:" << std::endl;
	for(int i = 0; i < N; ++i) {
		if (arr[i] != 1) {
			std::cout << "arr[" << i << "] expected " << 1 << ", got " << arr[i] << std::endl;
			errors++;
		}
	}


	std::cout << errors << " values were wrong" << std::endl;

	delete rs;
}
