/*
 * config.h
 *
 *  Created on: Nov 9, 2012
 *      Author: rr
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#define MAX_ITERATIONS 200

#define USE_GLUT

//#define USE_KDTREE
#define USE_HASHGRID

#define USE_GPU_HASH_GRID
//#define USE_GPU_MORTON_HASH_GRID
//#define USE_GPU_MORTON_GRID

#define MORTON_BITS 8

#define SORT_PHOTONHITS

//#define ENABLE_TIME_BREAKDOWN
#define NUM_THREADS 8

#define PHOTONS_PER_SLICE (1 << 20)

#define SM 15
#define FACTOR 256
#define BLOCKSIZE 512
#define BLOCKSIZE2D 16

#define EYEPASS_MAX_THREADS_PER_BLOCK 64
#define PHOTONPASS_MAX_THREADS_PER_BLOCK 64

#define MAX_EYE_PATH_DEPTH 16
#define MAX_PHOTON_PATH_DEPTH 8

// HACK should be MAX_PHOTON_PATH_DEPTH instead of 4
#define PHOTON_HIT_BUFFER_SIZE (PHOTONS_PER_SLICE * 4)

#define QBVH_STACK_SIZE 24

//#define WARP_RR

#endif /* CONFIG_H_ */
