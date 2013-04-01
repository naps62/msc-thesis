/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

#include <gamalib/gamalib.h>
#include <cstdlib>

#include <cublas.h>

#include "gama_ext/vector.h"
#include "kernel.h"

MemorySystem* LowLevelMemAllocator::_memSys = NULL;

#define N 100

//TODO add main function
int main() {
	RuntimeScheduler* rs = new RuntimeScheduler();

	// allocation
	int cpu_arr[N];
	gama::vector<int> arr(N);
	float alpha = 1.5f;

	// initialize with random data
	for(int i = 0; i < N; ++i) {
		cpu_arr[i] = i-1;
	}

	memcpy(arr.getPtr(), cpu_arr, sizeof(int) * N);
	rs->synchronize();
	rs->submit(new kernel(arr, 0, N));
	rs->synchronize();

	arr.resize(N*2);

	// initialize new values
	for(int i = N; i < N*2; ++i) {
		arr[i] = i;
	}

	rs->submit(new kernel(arr, 0, N*2));
	rs->synchronize();

	unsigned int errors = 0;

	for(int i = 0; i < N*2; ++i) {
		if (arr[i] != i + 1) {
			std::cout << "arr[" << i << "] expected " << i + 1 << ", got " << arr[i] << std::endl;
			errors++;
		}
	}
	std::cout << errors << " values were wrong" << " size:" << arr.size() << std::endl;

	delete rs;
}
