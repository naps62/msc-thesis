/*
 * memory_config.h
 *
 *  Created on: May 9, 2012
 *      Author: ricardo
 */

#ifndef MEMORY_CONFIG_H_
#define MEMORY_CONFIG_H_

const unsigned long MEM_BITS = 31l;		//<! Memory bits (determines memory size)
const unsigned long BLOCK_BITS = 12l;	//<! Block bits (determines block size) 1K
const unsigned long CACHE_BITS = 29l;	//<! Cache bits (determines cache size)
const unsigned long PAGE_BITS = BLOCK_BITS;

#define NUMBER_OF_HEAPS (24l+16l)

const unsigned long MEM_SIZE = (1l << MEM_BITS); 		 	//<! Global memory size
const unsigned long BLOCK_SIZE = (1l << BLOCK_BITS);	 	//<! Global block size
const unsigned long CACHE_SIZE = (1l << CACHE_BITS); 		//<! Device cache size

//Memory Blocks and Super Blocks
const unsigned long NUMBER_BLOCKS = MEM_SIZE >> BLOCK_BITS; 		 	//<! Total number of blocks
const unsigned long OFFSET_BITS = 6l; 									//<! Offset bits (determines number of blocks per super block)
const unsigned long BLOCK_PER_SUPER_BLOCK = (1l << OFFSET_BITS); 		//<! Number of blocks per Super Block (MAX is 64)
const unsigned long NUMBER_SUPER_BLOCKS= NUMBER_BLOCKS >> OFFSET_BITS; 	//<! Total number of Super Blocks

const unsigned long INVALID_REF = 0xFFFFFFFFFFFFFFFF; 	//<! Invalid reference
#define FULL_BLOCK 0xFFFFFFFFFFFFFFFF 					//<! All pages occupied (in a Super Block)



//Cache
const unsigned long CACHE_LINE = BLOCK_SIZE; 				 	//<! Cache line size
const unsigned long CACHE_INDEX_BITS = CACHE_BITS-BLOCK_BITS; 	//<! Total number of bits for index
const unsigned long CACHE_OFFSET_BITS = BLOCK_BITS; 			//<! Total number of bits for offset
const unsigned long NUMBER_PAGES = CACHE_SIZE >> PAGE_BITS;

const unsigned long address_filter = ~(BLOCK_SIZE-1);
//= ~((1 << CACHE_BITS) - 1); 		/*!< Address tag bit mask */
const unsigned long index_filter = (CACHE_SIZE - 1) & ~(BLOCK_SIZE-1); 	/*!< Address index bit mask */
const unsigned long offset_filter = (CACHE_SIZE - 1); /*!< Address offset bit mask */

typedef char VALID_FLAG;
const char CACHE_INVALID 	= 0;
const char CACHE_SHARED 	= 1;
const char CACHE_MODIFIED 	= 2;
const char CACHE_EXCLUSIVE 	= 4;

typedef char CACHE_TYPE;
const char CACHE_READ_ONLY 	= 0;
const char CACHE_READ_WRITE	= 1;


#endif /* MEMORY_CONFIG_H_ */
