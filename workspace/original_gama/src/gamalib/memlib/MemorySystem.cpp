/*
 * MemorySystem.cpp
 *
 *  Created on: May 9, 2012
 *      Author: ricardo
 */

//#include "MemorySystem.h"
//
//	/*!
//	 * @brief Memory System Constructor
//	 *
//	 * Creates the memory shared system for the entire application i.e. to be used by both CPU and CUDA enabled devices.
//	 * The method creates a pool of pinned memory with a specific size and a set of auxiliary structures to allow memory management.
//	 *
//	 * @note See documentation on CUDA 4.0 Unified Memory System
//	 * @note See memory_config.h
//	 *
//	 */
//	MemorySystem::MemorySystem() {
//		checkCudaErrors(cudaHostAlloc((void**) &MEM_POOL, MEM_SIZE, cudaHostAllocPortable)); 					// Allocates (pinned) memory for the Memory Pool
//		memset((unsigned long*) BLOCK_PAGES_USED_MASK, 0, sizeof(unsigned long) * NUMBER_SUPER_BLOCKS); 	// sets all memory super blocks pages to free (all pages available)
//		memset((unsigned long*) CONTIGUOUS_NUM_PAGES_ASSIGNED, 0, sizeof(unsigned long) * NUMBER_BLOCKS); 	// sets all memory super blocks pages to free (all pages available)
//		memset((unsigned char*) USED_SUPER_BLOCK, BLOCK_UNSED, sizeof(char) * NUMBER_SUPER_BLOCKS); 		// sets all memory super blocks to available - "BLOCK_UNSED" - Not assigned to any thread warp
//		memset((unsigned long*) HEAPS, INVALID_REF, sizeof(unsigned long) * (NUMBER_OF_HEAPS));
//
//		NEXT_SUPER_BLOCK[0]=1;
//		PREV_SUPER_BLOCK[NUMBER_SUPER_BLOCKS-1]=NUMBER_SUPER_BLOCKS-2;
//		NEXT_SUPER_BLOCK[NUMBER_SUPER_BLOCKS-1] = PREV_SUPER_BLOCK[0] = INVALID_REF;
//		for(unsigned int i=1; i< NUMBER_SUPER_BLOCKS-1; i++){
//			NEXT_SUPER_BLOCK[i]=i+1;
//			PREV_SUPER_BLOCK[i]=i-1;
//		}
//		FREE_SUPER_BLOCKS=0;
//
//	}
//
//	MemorySystem::~MemorySystem(){ }
//
//
//
//	/*! distruct
//	 * @brief Memory allocator
//	 *
//	 * Reserves memory in the system
//	 *
//	 * @param size Size (in bytes) to be allocated
//	 *
//	 * @return Returns a pointer to the beginning of the reserved memory location. In case of failure in allocation return a null pointer.
//	 */
//	__DEVICE__    __forceinline
//	void* allocate(size_t size) {
//		void* ret = NULL;
//		/*unsigned long superblock, block;
//
//		//printf("size %lu\n",size);
//		unsigned long nblocks = ceil(((double) size / BLOCK_SIZE));
//
//		// Fetches an Hiper Block if more than super block is necessary
//		if(nblocks>32) {
//			//printf("Flag Hyper %lu\n",nblocks);
//			unsigned long nsuperblocks = ceil( ((double) nblocks / (double)BLOCK_PER_SUPER_BLOCK) );
//			SIMDserial if (getHipperBlock(nsuperblocks,superblock)) {
//				ret = (void*) (MEM_POOL + ((unsigned long)(superblock*BLOCK_PER_SUPER_BLOCK) << BLOCK_BITS));
//			}
//		}
//
//		// If only a super block or less is necessary the "find_next_available_cell" method is called
//		else {
//			//printf("Flag Super %lu\n",nblocks);
//			SIMDserial if(find_next_available_cell(nblocks, superblock, block))
//				ret = (void*) (MEM_POOL + ((unsigned long)(superblock*BLOCK_PER_SUPER_BLOCK+block) << BLOCK_BITS));
//		}
//		return ret;*/
//	}
//
//
//
//	__DEVICE__    __forceinline
//	void deallocate(void* ptr) {
///*
//		unsigned long addr = ((unsigned long) ptr - (unsigned long) MEM_POOL)>> BLOCK_BITS; // Calculates the "virtual" address
//
//		unsigned long block_index = addr & (BLOCK_PER_SUPER_BLOCK - 1); // Calculates the page index
//		unsigned long super_block_index = addr >> OFFSET_BITS; // Calculates the super block index
//		unsigned long nblocks = CONTIGUOUS_NUM_PAGES_ASSIGNED[super_block_index	* BLOCK_PER_SUPER_BLOCK + block_index]; // Calculates the number of pages that the pointer represents
//
//		// If the pointer refers to a Hyper Block
//		if (USED_SUPER_BLOCK[super_block_index] == HYPER_BLOCK_ALLOCATED) {
//			//printf("Dealloc HYPER\n");
//			USED_SUPER_BLOCK[super_block_index] = BLOCK_UNSED;
//			unsigned long hiper_block_size = BLOCK_PAGES_USED_MASK[super_block_index];
//			GLOBAL_MEMORY_SYSTEM_LOCK.Acquire(warpID);
//			PREV_SUPER_BLOCK[FREE_SUPER_BLOCKS] = super_block_index+hiper_block_size-1; //set first free super block pointing to the last super block of the hyper block
//			NEXT_SUPER_BLOCK[super_block_index+hiper_block_size-1]=FREE_SUPER_BLOCKS;
//			FREE_SUPER_BLOCKS=super_block_index;
//			BLOCK_OCUPANCY[super_block_index] = 0;
//			GLOBAL_MEMORY_SYSTEM_LOCK.Release(warpID);
//			return;
//		}
//
//		//printf("Dealloc SUPER\n");
//		//Otherwise is a pointer to a (or group of) block(s) of a Super Block
//		int heapid = heapID; //falta a hash!!! todo
//
//		HEAP_LOCK[heapid].Acquire(warpID);
//		CONTIGUOUS_NUM_PAGES_ASSIGNED[super_block_index * BLOCK_PER_SUPER_BLOCK + block_index] = 0; // Sets the number of pages assigned for that pointer to 0;
//		unsigned long filter = ~(((1l << nblocks) - 1l) << block_index); // Calculates the filter of the pages to be freed from the super block
//		//unsigned long previous = BLOCK_PAGES_USED_MASK[super_block_index]; // Storing the current number of pages occupied
//
//		BLOCK_PAGES_USED_MASK[super_block_index] &= filter; // frees said pages from the respective block
//		BLOCK_OCUPANCY[super_block_index] -= nblocks;
//
//		// If the super block is now empty it must be removed from the heap and added to the empty super block pool
//		if (BLOCK_PAGES_USED_MASK[super_block_index] == 0) {
//			//remove from Heap
//			if(PREV_SUPER_BLOCK[super_block_index] == INVALID_REF) {
//				HEAPS[heapid] = NEXT_SUPER_BLOCK[super_block_index];
//				PREV_SUPER_BLOCK[HEAPS[heapid]] = INVALID_REF;
//
//			}
//			else {
//				NEXT_SUPER_BLOCK[PREV_SUPER_BLOCK[super_block_index]] = NEXT_SUPER_BLOCK[super_block_index];
//				PREV_SUPER_BLOCK[NEXT_SUPER_BLOCK[super_block_index]] = PREV_SUPER_BLOCK[super_block_index];
//			}
//
//			//Add to global super block pool
//			GLOBAL_MEMORY_SYSTEM_LOCK.Acquire(warpID);
//			NEXT_SUPER_BLOCK[super_block_index] = FREE_SUPER_BLOCKS;
//			PREV_SUPER_BLOCK[FREE_SUPER_BLOCKS] = super_block_index;
//			PREV_SUPER_BLOCK[super_block_index] = INVALID_REF;
//			FREE_SUPER_BLOCKS = super_block_index;
//			GLOBAL_MEMORY_SYSTEM_LOCK.Release(warpID);
//		}
//
//		HEAP_LOCK[heapid].Release(warpID);
//*/
//	}
//
//
//	/*!
//	 * @brief Custom new Operator
//	 *
//	 * Allocates the memory system in pinned memory for global use (CPU and CUDA devices)
//	 *
//	 * @param size Size (in bytes) of the MemorySystem
//	 * @return Returns the memory pointer to the MemorySystem Object
//	 */
//	void* operator new(size_t size) {
//		void* ptr = NULL;
//		checkCudaErrors(cudaHostAlloc((void**) &ptr, sizeof(MemorySystem),cudaHostAllocPortable));
//		return ptr;
//	}
//
//
//	/*!
//	 * @brief Custom delete Operator
//	 *
//	 * Frees the (pinned) memory of the MemorySystem Object
//	 *
//	 * @param ptr MemorySystem Object pointer to the location to be deallocated
//	 *
//	 */
//	void operator delete(void* ptr) {
//		checkCudaErrors(cudaFreeHost(ptr));
//	}

