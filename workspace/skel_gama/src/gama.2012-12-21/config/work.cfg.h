/*
 * work_config.h
 *
 *  Created on: Apr 26, 2011
 *      Author: jbarbosa
 */

#ifndef WORK_CONFIG_H_
#define WORK_CONFIG_H_



	/*!
	 * @brief No work identifier
	 */
	#define	W_NONE		0x00000000

	/*!
	 * @brief Bit mask for reserved bit in the identifier
	 * @note DO NOT CHANGE
	 */
	#define W_RESERVED_FLAGS	0xFF000000
	#define W_RESERVED_SIMD		0x00F00000
	#define W_RESERVED_CLASSIFY	0x00080000
	#define W_RESERVED			(W_RESERVED_FLAGS | W_RESERVED_SIMD | W_RESERVED_CLASSIFY)

	/**
	 * Work definitions moved to myconf/work.h
	 */
	#include <gamaconf/work.h>


	/*!
	 * @brief Job SIMD description
	 * Defines job descriptions.
	 * @note Only used by the CUDA device to determine groups of cooperating threads. The same
	 * job description will be copied (only pointer) with in the bucket.
	 * @note It will be stored in the WORK TYPE ID on the MSB.
	 */

	enum WORK_CLASSIFY {
		W_REGULAR 		= 0x00000000,
		W_IRREGULAR 	= 0x00080000
	};

	enum SIMD_TYPE {
		WIDE_SINGLE		= 0x00100000, /*!< NOT SIMD */
		WIDE_4 			= 0x00200000, /*!< 4-WIDE SIMD */
		WIDE_32			= 0x00400000, /*!< 32-WIDE SIMD */
		W_WIDE 			= 0x00800000
	};

	enum WORK_DESCRIPTION {
		WD_NONE			= 0x00000000,
		WD_NOT_DICEABLE = 0x01000000,
		WD_CALLBACK		= 0x02000000,
		WD_WAIT_BARRIER = 0x04000000,
		WD_SAMPLING 	= 0x08000000
	};

	enum SPECIAL_KERNEL {
		W_GENERAL 		= 0x00000000,
		W_SCAN 			= 0x10000000,
		W_REDUCE 		= 0x20000000
	};



	/**
	 * Reserved types and flags
	 */
	#define FILTER_WORK(W_T_I) (W_T_I & ~W_RESERVED)

	#define FILTER_SIMD(W_T_I) (W_T_I & W_RESERVED_SIMD)
	#define FILTER_FLAGS(W_T_I) (W_T_I & W_RESERVED_FLAGS)
	#define GET_SIMD(W_T_I) ((W_T_I & W_RESERVED_SIMD) >> 20l)

	#define IS_NOTDICEABLE(W_T_I) 	(W_T_I & WD_NOT_DICEABLE)
	#define IS_DICEABLE(W_T_I) 	((W_T_I & WD_NOT_DICEABLE) == 0)
	#define SWAP_DICEABLE(W_T_I) (W_T_I = (W_T_I ^ WD_NOT_DICEABLE))

	#define HAS_CALLBACK(W_T_I) (W_T_I & WD_CALLBACK)
	#define SWAP_CALLBACK(W_T_I) (W_T_I = (W_T_I ^ WD_CALLBACK))

	#define IS_WIDE_KERNEL(W_T_I) (W_T_I & W_WIDE)
	#define SWAP_WIDE_KERNEL(W_T_I) (W_T_I = (W_T_I ^ W_WIDE))

	#define IS_REDUCE_KERNEL(W_T_I) (W_T_I & W_REDUCE)
	#define SWAP_REDUCE_KERNEL(W_T_I) (W_T_I = (W_T_I ^ W_REDUCE))

	#define IS_SCAN_KERNEL(W_T_I) (W_T_I & W_SCAN)
	#define SWAP_SCAN_KERNEL(W_T_I) (W_T_I = (W_T_I ^ W_SCAN))

	#define IS_REGULAR(W_T_I) ((W_T_I & W_IRREGULAR) == 0)
	#define IS_IRREGULAR(W_T_I) (W_T_I & W_IRREGULAR)
	#define SWAP_CLASSIFY(W_T_I) (W_T_I = (W_T_I ^ W_IRREGULAR))

	#define IS_SAMPLING(W_T_I) (W_T_I & WD_SAMPLING)
	#define SWAP_SAMPLING(W_T_I) (W_T_I = (W_T_I ^ WD_SAMPLING))

#endif /* WORK_CONFIG_H_ */
