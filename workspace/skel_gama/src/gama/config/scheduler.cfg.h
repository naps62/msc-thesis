/*
 * scheduler.cfg.h
 *
 *  Created on: Sep 16, 2012
 *      Author: jbarbosa
 */

#ifndef SCHEDULER_CFG_H_
#define SCHEDULER_CFG_H_

// *********************************** SCHEDULING:

//#define STATIC
//#define DYNAMIC
//#define ADAPTIVE
//#define SSD

#ifdef STATIC
#define SC_SELECTED
#endif

#ifdef DYNAMIC
#define SC_SELECTED
#endif

#ifdef SSD
#define SC_SELECTED
#define DEMAND_DRIVEN
#endif

#ifndef SC_SELECTED
#define DEMAND_DRIVEN
#endif

#define SCHEDULER_CHUNK 8
#define SCHEDULER_MINIMUM 1

// *********************************** SCHEDULING;

#define warpID 1
#define heapID 1

#endif /* SCHEDULER_CFG_H_ */
