/*
 * config.h
 *
 *  Created on: Nov 9, 2012
 *      Author: rr
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#define DISABLE_TIME_BREAKDOWN

#define REBUILD_HASH

#define MAX_ITERATIONS 11

//#define USE_PPM
/**
 * PPM
 * Single device.
 * Dependant iterations, single build hitpoints, reduce radius and reflected flux.
 * Radius per iteration, dependant and per hit point.
 * Keep local statistics.
 */

//#define USE_SPPM
/**
 * SPPM
 * Single device.
 * Dependant iterations, in each iterations build hitpoints, reduce radius and reflected flux.
 * Radius per iteration, dependant and per hit point.
 * Keep local statistics.
 */

//#define USE_SPPMPA
/**
 * SPPM:PA
 * Each device builds hitpoints and hash.
 * Iterations independent, radius not reduced -> precalculated.
 * Radius per iteration, not per hitpoint.
 * 1 inital SPP.
 * Paper PPM:PA approach reversed.
 */

#define USE_PPMPA
/**
 * PPM:PA
 * Single hit points, each device mirrors hpts and builds hash grid.
 * Iterations independent, radius not reduced.
 * Oversampling.
 * Multi-resolution grid targeted.
 */

#ifdef DISABLE_TIME_BREAKDOWN
#define USE_GLUT
#endif

//#define RENDER_FAST_PHOTON
#define RENDER_TINY
//#define RENDER_MEDIUM
//#define RENDER_BIG
//#define RENDER_HUGE


//#define CPU
//#define GPU0
#define GPU2


#define SM 15
#define FACTOR 256
#define BLOCKSIZE 512

#define MAX_EYE_PATH_DEPTH 16
#define MAX_PHOTON_PATH_DEPTH 8

#define QBVH_STACK_SIZE 24

//#define WARP_RR

#endif /* CONFIG_H_ */
