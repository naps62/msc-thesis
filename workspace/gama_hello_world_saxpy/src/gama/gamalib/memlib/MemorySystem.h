/*
 * MemorySystem.h
 *
 *  Created on: May 9, 2012
 *      Author: ricardo
 */

#ifndef MEMORY_SYSTEM_H_
#define MEMORY_SYSTEM_H_

#include <config/common.h>
#include "SpinLock.h"


#include <gamalib/utils/GlobalMutex.h>
#include <gamalib/utils/cuda_utils.cuh>

class MemorySystem{


protected:

	/*!
	 * @enum BLOCK_STATUS
	 *
	 * @brief The three possible states of a memory Super Block
	 */
	enum BLOCK_STATUS {
		BLOCK_UNUSED = 0x0, 			//!< Super Block is assigned to a warp
		BLOCK_USED = 0x1, 				//!< Super Block is not assigned
		HYPER_BLOCK_ALLOCATED = 0xFF 	//!< Super block is assigned to a warp and is a block of an Hiper Block
	};


public:

	void* MEM_POOL;//[MEM_SIZE]; 									//!< Global Memory Pool
	//unsigned char USED_SUPER_BLOCK[NUMBER_BLOCKS];			//!< Super Blocks staus (Used, Unused or allocated as part of an Hyper-block)
	unsigned char USED_SUPER_BLOCK[NUMBER_SUPER_BLOCKS];
	unsigned long NEXT_SUPER_BLOCK[NUMBER_SUPER_BLOCKS];
	unsigned long PREV_SUPER_BLOCK[NUMBER_SUPER_BLOCKS];
	unsigned long FREE_SUPER_BLOCKS;
	unsigned long HEAPS[NUMBER_OF_HEAPS];
	unsigned long BLOCK_PAGES_USED_MASK[NUMBER_SUPER_BLOCKS]; 	//!< Contains the mask for pages assigned within the block (1-Assigned | 0-Unused). In the case of HyperBlock, stores the number of consecutive SuperBlocks
	unsigned long CONTIGUOUS_NUM_PAGES_ASSIGNED[NUMBER_BLOCKS]; //!< Assigned size in number of pages
	unsigned long BLOCK_OCUPANCY[NUMBER_SUPER_BLOCKS]; 			//!< Number of pages occupied per block

	unsigned long FREED_HYPER_BLOCKS;

	SpinLock GLOBAL_MEMORY_SYSTEM_LOCK;
	//GlobalMutex<16+4> GLOBAL_MEMORY_SYSTEM_LOCK;
	//HybridMutex<TOTAL_DEVICES> GLOBAL_MEMORY_SYSTEM_LOCK;
	SpinLock HEAP_LOCK[NUMBER_OF_HEAPS];
#if defined OCCUPANCY
	float occup_ratio;
#endif
	unsigned long teste;
	unsigned DEVICES_CORE_IDS_S[TOTAL_DEVICES];// = {0,1,2,18,19};

//#if defined(GAMA_CACHE)
//	DeviceCache* caches[TOTAL_DEVICES];
	unsigned long PAGES_LOCATION[NUMBER_BLOCKS];

//	__host__
//	bool pageMove(unsigned index, unsigned devID, W_NOCOPY* reason){
//		unsigned long* a = &(PAGES_LOCATION[index]);
//		unsigned long ret = __sync_val_compare_and_swap((volatile unsigned long*)a,INVALID_REF,devID);
//		if(ret==INVALID_REF) return true;
//		(ret==devID)?(*reason=W_PAGE_VALID):(*reason=W_PAGE_BUSY);
//		return false;
//	}
//
//	__host__
//	bool pageMoveBack(unsigned index, unsigned devID){
//		unsigned long* a = &(PAGES_LOCATION[index]);
//		unsigned long ret = __sync_val_compare_and_swap((volatile unsigned long*)a,devID,INVALID_REF);
//		return true;
//	}

//#endif


public:

	MemorySystem(){
		(cudaHostAlloc((void**) &MEM_POOL, MEM_SIZE, cudaHostAllocPortable)); 					// Allocates (pinned) memory for the Memory Pool
		memset((unsigned long*) BLOCK_PAGES_USED_MASK, 0, sizeof(unsigned long) * NUMBER_SUPER_BLOCKS); 	// sets all memory super blocks pages to free (all pages available)
		memset((unsigned long*) CONTIGUOUS_NUM_PAGES_ASSIGNED, 0, sizeof(unsigned long) * NUMBER_BLOCKS); 	// sets all memory super blocks pages to free (all pages available)
		memset((unsigned char*) USED_SUPER_BLOCK, BLOCK_UNUSED, sizeof(char) * NUMBER_SUPER_BLOCKS); 		// sets all memory super blocks to available - "BLOCK_UNUSED" - Not assigned to any thread warp
		memset((unsigned long*) HEAPS, 0xFF, sizeof(unsigned long) * (NUMBER_OF_HEAPS));
		memset((unsigned long*) PAGES_LOCATION, 0xFF, sizeof(unsigned long) * NUMBER_BLOCKS);

		GLOBAL_MEMORY_SYSTEM_LOCK = SpinLock();

		NEXT_SUPER_BLOCK[0]=1;
		PREV_SUPER_BLOCK[NUMBER_SUPER_BLOCKS-1] = NUMBER_SUPER_BLOCKS-2;
		NEXT_SUPER_BLOCK[NUMBER_SUPER_BLOCKS-1] = PREV_SUPER_BLOCK[0] = INVALID_REF;
		for(unsigned int i=1; i< NUMBER_SUPER_BLOCKS-1; i++){
			NEXT_SUPER_BLOCK[i]=i+1;
			PREV_SUPER_BLOCK[i]=i-1;
		}
		FREE_SUPER_BLOCKS=0;
		FREED_HYPER_BLOCKS=INVALID_REF;

		//printf("Memeory System initiated\n");
		teste=1234567890;

		//printf("{ 0");
		DEVICES_CORE_IDS_S[0]=0;
		for(int i=1; i<TOTAL_DEVICES; i++){
			DEVICES_CORE_IDS_S[i] = DEVICES_CORE_IDS_S[i-1] + TOTAL_CORES_TYPES[i-1];
			//printf(" %d",DEVICES_CORE_IDS_S[i]);
		}
		//printf(" }\n");

		//for(int i=0; i< NUMBER_OF_HEAPS; i++) HEAP_LOCK[i]=SpinLock();
		//printf("size of long long int %lu\n",sizeof(long long int));
	}

	~MemorySystem(){}



private:

	__DEVICE__  __forceinline
	unsigned long getHeapID(unsigned int coreID){
		unsigned msig = coreID >> 16u;
		unsigned lsig = coreID & ((1<<16u)-1);
		unsigned desl = DEVICES_CORE_IDS_S[msig];
		return desl+lsig;

	}

	__DEVICE__    __forceinline
	int countFreeSuperBlocks(){
		unsigned long it = FREE_SUPER_BLOCKS;
		int count=0;
		while(it != INVALID_REF){
			count++;
			it = NEXT_SUPER_BLOCK[it];
		}
		return count;
	}

	__DEVICE__    __forceinline
	bool getNewSuperBlock(unsigned long &superblock){
		GLOBAL_MEMORY_SYSTEM_LOCK.Acquire(CORE_ID); //todo warpID não serve. Tem de ser outra coisa!!
		//superblock = INVALID_REF;

		superblock = FREED_HYPER_BLOCKS;
		if (superblock!=INVALID_REF){
			unsigned long hyper_block_size = BLOCK_PAGES_USED_MASK[FREED_HYPER_BLOCKS];
			if (hyper_block_size>1) BLOCK_PAGES_USED_MASK[superblock+1] = BLOCK_PAGES_USED_MASK[superblock]-1;
			FREED_HYPER_BLOCKS = NEXT_SUPER_BLOCK[superblock]; //can't be FREED_HYPER_BLOCKS++;
			NEXT_SUPER_BLOCK[superblock]=INVALID_REF;
			PREV_SUPER_BLOCK[superblock]=INVALID_REF;
			BLOCK_OCUPANCY[superblock]=0;

			USED_SUPER_BLOCK[superblock]=BLOCK_UNUSED;
			BLOCK_PAGES_USED_MASK[superblock]=0;
		}

		if (superblock==INVALID_REF && FREE_SUPER_BLOCKS!=INVALID_REF){
			superblock = FREE_SUPER_BLOCKS;
			//if (superblock!=INVALID_REF){
			FREE_SUPER_BLOCKS = NEXT_SUPER_BLOCK[superblock];
			PREV_SUPER_BLOCK[superblock]=INVALID_REF;
			PREV_SUPER_BLOCK[FREE_SUPER_BLOCKS] = INVALID_REF;
			BLOCK_OCUPANCY[superblock]=0;
			//}
		}

		GLOBAL_MEMORY_SYSTEM_LOCK.Release(CORE_ID); //todo warpID não serve. Tem de ser outra coisa!!
		return (superblock!=INVALID_REF);
	}


	__DEVICE__    __forceinline
	bool getHipperBlockFromFreedHyperBlocks(unsigned long nsuperblocks, unsigned long &superblock) {
		//printf("Entrei aqui 2!!!!!!!\n");
		unsigned long temp = FREED_HYPER_BLOCKS;
		while(temp!=INVALID_REF){
			if(BLOCK_PAGES_USED_MASK[temp]==nsuperblocks){
				FREED_HYPER_BLOCKS = temp+nsuperblocks;
				superblock = temp;
				temp=INVALID_REF;
			}
			else {
				if(BLOCK_PAGES_USED_MASK[temp]>nsuperblocks){
					BLOCK_PAGES_USED_MASK[temp+nsuperblocks]-=nsuperblocks;
					BLOCK_PAGES_USED_MASK[temp]=nsuperblocks;
					//NEXT_SUPER_BLOCK[temp+nsuperblocks-1]=INVALID_REF;
					FREED_HYPER_BLOCKS = NEXT_SUPER_BLOCK[temp+nsuperblocks-1];

					}
				else temp = NEXT_SUPER_BLOCK[temp+BLOCK_PAGES_USED_MASK[temp]];
			}
		}

		return superblock!=INVALID_REF;
	}

	__DEVICE__    __forceinline
	bool getHipperBlock(unsigned long nsuperblocks, unsigned long &superblock) {

		//if(getHipperBlockFromFreedHyperBlocks(nsuperblocks, superblock)) return true;

		GLOBAL_MEMORY_SYSTEM_LOCK.Acquire(CORE_ID);

		unsigned long indx = FREE_SUPER_BLOCKS;
		unsigned long tmp = FREE_SUPER_BLOCKS;
		//int aux = countFreeSuperBlocks();
		unsigned long count=1;
		//if(indx!=INVALID_REF) count++;
		while(indx!=INVALID_REF && count<nsuperblocks){
			//if(count==0) tmp=indx;
			if(NEXT_SUPER_BLOCK[indx]==indx+1) {count++; indx++;}
			else{
				count=0;
				indx=NEXT_SUPER_BLOCK[indx];
				tmp=indx;
			}
		}

		//printf("tmp=%lu \t indx=%lu (%lu)\n",tmp,indx,indx-tmp+1);
		if(count==nsuperblocks && indx!=INVALID_REF){
			//printf("Encontrei um hyper block\n");
			//printf("count=%lu nsuperblocks=%lu\n",count,nsuperblocks);
			if(PREV_SUPER_BLOCK[tmp]==INVALID_REF) {
				FREE_SUPER_BLOCKS=NEXT_SUPER_BLOCK[indx];
				PREV_SUPER_BLOCK[FREE_SUPER_BLOCKS] = INVALID_REF;
			}
			else NEXT_SUPER_BLOCK[PREV_SUPER_BLOCK[tmp]] = NEXT_SUPER_BLOCK[indx];
			USED_SUPER_BLOCK[tmp]=HYPER_BLOCK_ALLOCATED;
			BLOCK_PAGES_USED_MASK[tmp]=nsuperblocks;
			superblock=tmp;
			NEXT_SUPER_BLOCK[indx]=INVALID_REF;
		}

		else getHipperBlockFromFreedHyperBlocks(nsuperblocks, superblock);

		GLOBAL_MEMORY_SYSTEM_LOCK.Release(CORE_ID);
		//printf("Super_Blocks=%d\t nsuperblocks=%lu count=%lu \tSuper_Blocks=%d\n\n",aux, nsuperblocks, count, countFreeSuperBlocks());

		return (count==nsuperblocks);

	}

	__DEVICE__    __forceinline
	bool getBlock(unsigned long superblock, unsigned long nblocks, unsigned long &block) {
		if(BLOCK_PER_SUPER_BLOCK-BLOCK_OCUPANCY[superblock]<=nblocks) return false;
		block = INVALID_REF;
		unsigned long B_Mask = BLOCK_PAGES_USED_MASK[superblock];

		unsigned long Mask = (nblocks <= 1) ? 1ul : (1ul << nblocks) - 1; 	// Sets the mask with the number of contiguous blocks necessary
		unsigned long max_it = BLOCK_PER_SUPER_BLOCK - (nblocks - 1); 		// Maximum number of tries to fit the new mask in the super block


		unsigned long it = 0;

		while (it < max_it && block > 64) { // Since an unsigned long is 64 bits long, no super block can or will have more than 64 pages.

			if ((B_Mask & Mask) == 0) {
				BLOCK_PAGES_USED_MASK[superblock] |= Mask;
				block = it;
				CONTIGUOUS_NUM_PAGES_ASSIGNED[superblock*BLOCK_PER_SUPER_BLOCK+block] = nblocks;
				BLOCK_OCUPANCY[superblock]+=nblocks;
				return true;
			}
			Mask = Mask << 1;
			it++;
		}
		return false;//(block < 64);

	}

	__DEVICE__    __forceinline
	bool find_next_available_cell(unsigned long nblocks, unsigned long &superblock, unsigned long &block) {
		int heapid = getHeapID(CORE_ID); //falta a hash!!! todo
		unsigned long superblock_tmp;
		superblock=INVALID_REF;

		HEAP_LOCK[heapid].Acquire(heapid);//todo warpID não serve. Tem de ser outra coisa!!
		//Search for a Super block in the heap where the number of blocks requested fits
		superblock_tmp = HEAPS[heapid];
		while(superblock_tmp!=INVALID_REF){
			if(getBlock(superblock_tmp, nblocks, block)){ //{HEAP_LOCK[heapid].Release(warpID); return true;}
				superblock = superblock_tmp;
				break;
			}
			else superblock_tmp = NEXT_SUPER_BLOCK[superblock_tmp];
		}

		//if(superblock==INVALID_REF) printf("No Super block was found\n");
		//If no large enough Super block was found a new (empty) one is fetched from memory
		if(superblock==INVALID_REF){
			if (getNewSuperBlock(superblock)){
				NEXT_SUPER_BLOCK[superblock] = HEAPS[heapid];
				HEAPS[heapid]=superblock;
				getBlock(superblock, nblocks, block);
			}
		}


		HEAP_LOCK[heapid].Release(heapid);
		return (superblock!=INVALID_REF);
	}



public:

	__DEVICE__    __forceinline
	void* allocate(size_t size){
		//printf("Allocate size:%lu\n",size);
		void* ret = NULL;
		unsigned long superblock, block;

		unsigned long nblocks = ceil(((double) size / BLOCK_SIZE));

		// Fetches an Hyper Block if more than super block is necessary
		if(nblocks>32) {
			unsigned long nsuperblocks = ceil( ((double) nblocks / (double)BLOCK_PER_SUPER_BLOCK) );
			SIMDserial if (getHipperBlock(nsuperblocks,superblock)) {
				ret = (void*) ((unsigned long)MEM_POOL + ((unsigned long)(superblock*BLOCK_PER_SUPER_BLOCK) << BLOCK_BITS));
			}
			else printf("Error allocating!!!!!\n");
		}

		// If only a super block or less is necessary the "find_next_available_cell" method is called
		else {
			SIMDserial if(find_next_available_cell(nblocks, superblock, block))
				ret = (void*) ((unsigned long)MEM_POOL + ((unsigned long)(superblock*BLOCK_PER_SUPER_BLOCK+block) << BLOCK_BITS));
		}
		//printf("Allocationg Mem_pooll 0x%lx desl 0x%lx ret %lx !!\n",MEM_POOL, ((unsigned long)(superblock*BLOCK_PER_SUPER_BLOCK+block) << BLOCK_BITS), ret);
		return ret;
	}


	__DEVICE__    __forceinline
	void deallocate(void* ptr){
		unsigned long addr = ((unsigned long) ptr - (unsigned long) MEM_POOL)>> BLOCK_BITS; // Calculates the "virtual" address

		unsigned long block_index = addr & (BLOCK_PER_SUPER_BLOCK - 1); // Calculates the page index
		unsigned long super_block_index = addr >> OFFSET_BITS; // Calculates the super block index
		unsigned long nblocks = CONTIGUOUS_NUM_PAGES_ASSIGNED[super_block_index	* BLOCK_PER_SUPER_BLOCK + block_index]; // Calculates the number of pages that the pointer represents

		// If the pointer refers to a Hyper Block
		if (USED_SUPER_BLOCK[super_block_index] == HYPER_BLOCK_ALLOCATED) {
			//printf("Dealloc HYPER\n");
			USED_SUPER_BLOCK[super_block_index] = BLOCK_UNUSED;
			unsigned long hiper_block_size = BLOCK_PAGES_USED_MASK[super_block_index];
			GLOBAL_MEMORY_SYSTEM_LOCK.Acquire(CORE_ID);
			SIMDserial {
#if defined OCCUPANCY
			float size_r = float(hiper_block_size*BLOCK_PER_SUPER_BLOCK*BLOCK_SIZE*1.f)/float(MEM_SIZE); // Occupancy ratio
			occup_ratio-=size_r;
#endif
			if(FREED_HYPER_BLOCKS!=INVALID_REF) PREV_SUPER_BLOCK[FREED_HYPER_BLOCKS] = super_block_index+hiper_block_size-1; //set first free super block (of the hyper block) pointing to the last freed hyper block
			NEXT_SUPER_BLOCK[super_block_index+hiper_block_size-1]=FREED_HYPER_BLOCKS;
			FREED_HYPER_BLOCKS=super_block_index;
			BLOCK_OCUPANCY[super_block_index] = 0;
			}
			GLOBAL_MEMORY_SYSTEM_LOCK.Release(CORE_ID);
			return;
		}

		//printf("Dealloc SUPER\n");
		//Otherwise is a pointer to a (or group of) block(s) of a Super Block
		int heapid = getHeapID(CORE_ID); //falta a hash!!! todo

		HEAP_LOCK[heapid].Acquire(heapid);
		SIMDserial {
#if defined OCCUPANCY
		float size_r = float((CONTIGUOUS_NUM_PAGES_ASSIGNED[super_block_index * BLOCK_PER_SUPER_BLOCK + block_index])*BLOCK_SIZE*1.f)/float(MEM_SIZE); // Occupancy ratio
		occup_ratio-=size_r;
#endif

		CONTIGUOUS_NUM_PAGES_ASSIGNED[super_block_index * BLOCK_PER_SUPER_BLOCK + block_index] = 0; // Sets the number of pages assigned for that pointer to 0;
		unsigned long filter = ~(((1l << nblocks) - 1l) << block_index); // Calculates the filter of the pages to be freed from the super block
		//unsigned long previous = BLOCK_PAGES_USED_MASK[super_block_index]; // Storing the current number of pages occupied

		BLOCK_PAGES_USED_MASK[super_block_index] &= filter; // frees said pages from the respective block
		BLOCK_OCUPANCY[super_block_index] -= nblocks;

		// If the super block is now empty it must be removed from the heap and added to the empty super block pool
		if (BLOCK_PAGES_USED_MASK[super_block_index] == 0) {
			//remove from Heap
			if(PREV_SUPER_BLOCK[super_block_index] == INVALID_REF) {
				HEAPS[heapid] = NEXT_SUPER_BLOCK[super_block_index];
				PREV_SUPER_BLOCK[HEAPS[heapid]] = INVALID_REF;

			}
			else {
				NEXT_SUPER_BLOCK[PREV_SUPER_BLOCK[super_block_index]] = NEXT_SUPER_BLOCK[super_block_index];
				PREV_SUPER_BLOCK[NEXT_SUPER_BLOCK[super_block_index]] = PREV_SUPER_BLOCK[super_block_index];
			}

			//Add to global super block pool
			GLOBAL_MEMORY_SYSTEM_LOCK.Acquire(CORE_ID);
			NEXT_SUPER_BLOCK[super_block_index] = FREE_SUPER_BLOCKS;
			PREV_SUPER_BLOCK[FREE_SUPER_BLOCKS] = super_block_index;
			PREV_SUPER_BLOCK[super_block_index] = INVALID_REF;
			FREE_SUPER_BLOCKS = super_block_index;
			GLOBAL_MEMORY_SYSTEM_LOCK.Release(CORE_ID);
		}
		}

		HEAP_LOCK[heapid].Release(heapid);

	}




	void* operator new(size_t size){
		printf("Memory System allocated, %lu\n",sizeof(MemorySystem));
		void* ptr = NULL;
		(cudaHostAlloc((void**) &ptr, sizeof(MemorySystem),cudaHostAllocPortable));
		return ptr;
	}

	void operator delete(void* ptr){
		printf("Memory System de-allocated\n");
		(cudaFreeHost(ptr));
	}



	//APAGAR!!!!!
/*==============================================================================================================*/


//	__DEVICE__    __forceinline
//	bool find_next_available_cell2(unsigned long nblocks, unsigned long &superblock, unsigned long &block) {
//		int heapid = getHeapID(CORE_ID);
//		unsigned long superblock_tmp;
//		superblock=INVALID_REF;
//		printf("TID %lu CORE_ID %u heapid %d\n", TID, CORE_ID, heapid);
//		HEAP_LOCK[heapid].Acquire(heapid);//todo warpID não serve. Tem de ser outra coisa!!
////		//Search for a Super block in the heap where the number of blocks requested fits
////		superblock_tmp = HEAPS[heapid];
////		while(superblock_tmp!=INVALID_REF){
////			if(getBlock(superblock_tmp, nblocks, block)){
////				superblock = superblock_tmp;
////				break;
////			}
////			else superblock_tmp = NEXT_SUPER_BLOCK[superblock_tmp];
////		}
////
////		//if(superblock==INVALID_REF) printf("No Super block was found\n");
////		//If no large enough Super block was found a new (empty) one is fetched from memory
////		if(superblock==INVALID_REF){
////			if (getNewSuperBlock(superblock)){
////				NEXT_SUPER_BLOCK[superblock] = HEAPS[heapid];
////				HEAPS[heapid]=superblock;
////				getBlock(superblock, nblocks, block);
////			}
////		}
//
//
//		HEAP_LOCK[heapid].Release(heapid);
//		return (superblock!=INVALID_REF);
//	}
//
//
//	__DEVICE__    __forceinline
//	void* allocate2(size_t size){
//		//printf("Allocate size:%lu\n",size);
//		void* ret = NULL;
//		unsigned long superblock, block;
//
//		unsigned long nblocks = ceil(((double) size / BLOCK_SIZE));
//
//		// Fetches an Hyper Block if more than super block is necessary
//		if(nblocks>32) {
//			unsigned long nsuperblocks = ceil( ((double) nblocks / (double)BLOCK_PER_SUPER_BLOCK) );
//			SIMDserial if (getHipperBlock(nsuperblocks,superblock)) {
//				ret = (void*) (MEM_POOL + ((unsigned long)(superblock*BLOCK_PER_SUPER_BLOCK) << BLOCK_BITS));
//			}
//			else printf("Error allocating!!!!!\n");
//		}
//
//		// If only a super block or less is necessary the "find_next_available_cell" method is called
//		else {
//			SIMDserial if(find_next_available_cell2(nblocks, superblock, block))
//				ret = (void*) (MEM_POOL + ((unsigned long)(superblock*BLOCK_PER_SUPER_BLOCK+block) << BLOCK_BITS));
//		}
//		//printf("%u finished allocation!!\n",size);
//		return ret;
//	}
//
//
//	__DEVICE__    __forceinline
//		void deallocate2(void* ptr){
//		printf("ptr=%lx\n",ptr);
//			unsigned long addr = ((unsigned long) ptr - (unsigned long) MEM_POOL)>> BLOCK_BITS; // Calculates the "virtual" address
//
//			unsigned long block_index = addr & (BLOCK_PER_SUPER_BLOCK - 1); // Calculates the page index
//			unsigned long super_block_index = addr >> OFFSET_BITS; // Calculates the super block index
//			unsigned long nblocks = CONTIGUOUS_NUM_PAGES_ASSIGNED[super_block_index	* BLOCK_PER_SUPER_BLOCK + block_index]; // Calculates the number of pages that the pointer represents
//
//			//if(super_block_index >=NUMBER_SUPER_BLOCKS) printf("Está qui o problema %lu!!\n",super_block_index);
//			//printf("%c\n",USED_SUPER_BLOCK[super_block_index]);
//			// If the pointer refers to a Hyper Block
//			if (false/*USED_SUPER_BLOCK[super_block_index] == HYPER_BLOCK_ALLOCATED*/) {
//				printf("Dealloc HYPER\n");
////				USED_SUPER_BLOCK[super_block_index] = BLOCK_UNUSED;
////				unsigned long hiper_block_size = BLOCK_PAGES_USED_MASK[super_block_index];
////				GLOBAL_MEMORY_SYSTEM_LOCK.Acquire(CORE_ID);
////				SIMDserial {
////	#if defined OCCUPANCY
////				float size_r = float(hiper_block_size*BLOCK_PER_SUPER_BLOCK*BLOCK_SIZE*1.f)/float(MEM_SIZE); // Occupancy ratio
////				occup_ratio-=size_r;
////	#endif
////				if(FREED_HYPER_BLOCKS!=INVALID_REF) PREV_SUPER_BLOCK[FREED_HYPER_BLOCKS] = super_block_index+hiper_block_size-1; //set first free super block (of the hyper block) pointing to the last freed hyper block
////				NEXT_SUPER_BLOCK[super_block_index+hiper_block_size-1]=FREED_HYPER_BLOCKS;
////				FREED_HYPER_BLOCKS=super_block_index;
////				BLOCK_OCUPANCY[super_block_index] = 0;
////				}
////				GLOBAL_MEMORY_SYSTEM_LOCK.Release(CORE_ID);
////				return;
//			}
////
////			//printf("Dealloc SUPER\n");
////			//Otherwise is a pointer to a (or group of) block(s) of a Super Block
////			int heapid = getHeapID(CORE_ID); //falta a hash!!! todo
////
////			HEAP_LOCK[heapid].Acquire(heapid);
////			SIMDserial {
////	#if defined OCCUPANCY
////			float size_r = float((CONTIGUOUS_NUM_PAGES_ASSIGNED[super_block_index * BLOCK_PER_SUPER_BLOCK + block_index])*BLOCK_SIZE*1.f)/float(MEM_SIZE); // Occupancy ratio
////			occup_ratio-=size_r;
////	#endif
////
////			CONTIGUOUS_NUM_PAGES_ASSIGNED[super_block_index * BLOCK_PER_SUPER_BLOCK + block_index] = 0; // Sets the number of pages assigned for that pointer to 0;
////			unsigned long filter = ~(((1l << nblocks) - 1l) << block_index); // Calculates the filter of the pages to be freed from the super block
////			//unsigned long previous = BLOCK_PAGES_USED_MASK[super_block_index]; // Storing the current number of pages occupied
////
////			BLOCK_PAGES_USED_MASK[super_block_index] &= filter; // frees said pages from the respective block
////			BLOCK_OCUPANCY[super_block_index] -= nblocks;
////
////			// If the super block is now empty it must be removed from the heap and added to the empty super block pool
////			if (BLOCK_PAGES_USED_MASK[super_block_index] == 0) {
////				//remove from Heap
////				if(PREV_SUPER_BLOCK[super_block_index] == INVALID_REF) {
////					HEAPS[heapid] = NEXT_SUPER_BLOCK[super_block_index];
////					PREV_SUPER_BLOCK[HEAPS[heapid]] = INVALID_REF;
////
////				}
////				else {
////					NEXT_SUPER_BLOCK[PREV_SUPER_BLOCK[super_block_index]] = NEXT_SUPER_BLOCK[super_block_index];
////					PREV_SUPER_BLOCK[NEXT_SUPER_BLOCK[super_block_index]] = PREV_SUPER_BLOCK[super_block_index];
////				}
////
////				//Add to global super block pool
////				GLOBAL_MEMORY_SYSTEM_LOCK.Acquire(CORE_ID);
////				NEXT_SUPER_BLOCK[super_block_index] = FREE_SUPER_BLOCKS;
////				PREV_SUPER_BLOCK[FREE_SUPER_BLOCKS] = super_block_index;
////				PREV_SUPER_BLOCK[super_block_index] = INVALID_REF;
////				FREE_SUPER_BLOCKS = super_block_index;
////				GLOBAL_MEMORY_SYSTEM_LOCK.Release(CORE_ID);
////			}
////			}
////
////			HEAP_LOCK[heapid].Release(heapid);
//
//		}

/*==============================================================================================================*/

};


#endif //MEMORY_SYSTEM_H_
