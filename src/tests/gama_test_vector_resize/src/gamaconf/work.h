/*
 * work.h
 *
 *  Created on: Dec 14, 2012
 *      Author: naps62
 */

#ifndef __MY_WORK_H_
#define __MY_WORK_H_

#include <config/work.cfg.h>

/*!
 * @brief Work type identification
 * @note DO NOT CHANGE W_NONE or W_RESERVED
 */
enum WORK_TYPE {
	WORK_NONE = W_NONE, 	   /*!< Empty job description */

	WORK_KERNEL,

	WORK_TOTAL, 			   /*!< TOTAL NUMBER OF JOB DEFINTIONS */
	WORK_RESERVED = W_RESERVED /*!< RESERVED BIT MASK JOB */
	};

#endif // __MY_WORK_H_
