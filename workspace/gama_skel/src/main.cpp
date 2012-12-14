/*
 * main.cpp
 *
 *  Created on: December 14, 2012
 *      Author: Miguel Palhas
 */

#include <gama.h>

MemorySystem* LowLevelMemAllocator::_memSys = NULL;

//TODO add main function
int main() {
	RuntimeScheduler* rs = new RuntimeScheduler();

	cout << "oh look. it works!" << endl;
}
