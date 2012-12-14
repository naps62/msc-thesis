/*
 CUDA BarnesHut v2.2: Simulation of the gravitational forces
 in a galactic cluster using the Barnes-Hut n-body algorithm

 Copyright (c) 2011, Texas State University-San Marcos.  All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of Texas State University-San Marcos nor the names of its
 contributors may be used to endorse or promote products derived from this
 software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 OF THE POSSIBILITY OF SUCH DAMAGE.

 Author: Martin Burtscher
 */

#if (SAMPLE==5)

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#include <config/common.h>
#include <gamalib/gamalib.h>
#include <gamalib/gamalib.cuh>

#include "bh-martin.h"
#include <samples/BH/BHForce.h>
#include "createForce.h"

#include <samples/MB-NBODY/display.h>


// thread count
#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 1024
#define THREADS4 256
#define THREADS5 256
#define THREADS6 512

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 1
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 1  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 3

#define WARPSIZE 32
#define MAXDEPTH 32

/******************************************************************************/

// childd is aliased with velxd, velyd, velzd, accxd, accyd, acczd, and sortd but they never use the same memory locations
__constant__ int nnodesd, nbodiesd;
__constant__ float dtimed, dthfd, epssqd, itolsqd;
__constant__ volatile float *massd, *posxd, *posyd, *poszd, *velxd, *velyd, *velzd, *accxd, *accyd, *acczd;
__constant__ volatile float *maxxd, *maxyd, *maxzd, *minxd, *minyd, *minzd;
__constant__ volatile int *errd, *sortd, *childd, *countd, *startd;
__constant__ int *count_iterations;
__device__ volatile int stepd, bottomd, maxdepthd, blkcntd;
__device__ volatile float radiusd;


/******************************************************************************/
/*** Timing vars       ********************************************************/
/******************************************************************************/

cudaEvent_t start, stop;
clock_t starttime, endtime;
float timing[7];
int blocks;



float* m;
float* px;
float* py;
float* pz;
float* vx;
float* vy;
float* vz;
float* ax;
float* ay;
float* az;
int* ch;
int* st;

unsigned int step=0;

unsigned int nbodies,nnodes;

unsigned int* citl;

RuntimeScheduler *rs;

float* dq;
/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/

__global__ void InitializationKernelGPU() {
	*errd = 0;
	stepd = -1;
	maxdepthd = 1;
	blkcntd = 0;
}

/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS1, FACTOR1)
void BoundingBoxKernelGPU() {
	register int i, j, k, inc;
	register float val, minx, maxx, miny, maxy, minz, maxz;
	__shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1],
			smaxy[THREADS1], sminz[THREADS1], smaxz[THREADS1];

	// initialize with valid data (in case #bodies < #threads)
	minx = maxx = posxd[0];
	miny = maxy = posyd[0];
	minz = maxz = poszd[0];

	// scan all bodies
	i = threadIdx.x;
	inc = THREADS1 * gridDim.x;
	for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
		val = posxd[j];
		minx = min(minx, val);
		maxx = max(maxx, val);
		val = posyd[j];
		miny = min(miny, val);
		maxy = max(maxy, val);
		val = poszd[j];
		minz = min(minz, val);
		maxz = max(maxz, val);
	}

	// reduction in shared memory
	sminx[i] = minx;
	smaxx[i] = maxx;
	sminy[i] = miny;
	smaxy[i] = maxy;
	sminz[i] = minz;
	smaxz[i] = maxz;

	for (j = THREADS1 / 2; j > 0; j /= 2) {
		__syncthreads();
		if (i < j) {
			k = i + j;
			sminx[i] = minx = min(minx, sminx[k]);
			smaxx[i] = maxx = max(maxx, smaxx[k]);
			sminy[i] = miny = min(miny, sminy[k]);
			smaxy[i] = maxy = max(maxy, smaxy[k]);
			sminz[i] = minz = min(minz, sminz[k]);
			smaxz[i] = maxz = max(maxz, smaxz[k]);
		}
	}

	// write block result to global memory
	if (i == 0) {
		k = blockIdx.x;
		minxd[k] = minx;
		maxxd[k] = maxx;
		minyd[k] = miny;
		maxyd[k] = maxy;
		minzd[k] = minz;
		maxzd[k] = maxz;

		inc = gridDim.x - 1;
		if (inc == atomicInc((unsigned int *) &blkcntd, inc)) {
			// I'm the last block, so combine all block results
			for (j = 0; j <= inc; j++) {
				minx = min(minx, minxd[j]);
				maxx = max(maxx, maxxd[j]);
				miny = min(miny, minyd[j]);
				maxy = max(maxy, maxyd[j]);
				minz = min(minz, minzd[j]);
				maxz = max(maxz, maxzd[j]);
			}

			// compute 'radius'
			val = max(maxx - minx, maxy - miny);
			radiusd = max(val, maxz - minz) * 0.5f;

			// create root node
			k = nnodesd;
			bottomd = k;

			massd[k] = -1.0f;
			startd[k] = 0;
			posxd[k] = (minx + maxx) * 0.5f;
			posyd[k] = (miny + maxy) * 0.5f;
			poszd[k] = (minz + maxz) * 0.5f;
			k *= 8;
			for (i = 0; i < 8; i++)
				childd[k + i] = -1;

			stepd++;
		}
	}
}

/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernelGPU() {
	register int i, j, k, depth, localmaxdepth, skip, inc;
	register float x, y, z, r;
	register float px, py, pz;
	register int ch, n, cell, locked, patch;
	register float radius, rootx, rooty, rootz;

	// cache root data
	radius = radiusd;
	rootx = posxd[nnodesd];
	rooty = posyd[nnodesd];
	rootz = poszd[nnodesd];

	localmaxdepth = 1;
	skip = 1;
	inc = blockDim.x * gridDim.x;
	i = threadIdx.x + blockIdx.x * blockDim.x;

	// iterate over all bodies assigned to thread
	while (i < nbodiesd) {
		if (skip != 0) {
			// new body, so start traversing at root
			skip = 0;
			px = posxd[i];
			py = posyd[i];
			pz = poszd[i];
			n = nnodesd;
			depth = 1;
			r = radius;
			j = 0;
			// determine which child to follow
			if (rootx < px)
				j = 1;
			if (rooty < py)
				j += 2;
			if (rootz < pz)
				j += 4;
		}

		// follow path to leaf cell
		ch = childd[n * 8 + j];
		while (ch >= nbodiesd) {
			n = ch;
			depth++;
			r *= 0.5f;
			j = 0;
			// determine which child to follow
			if (posxd[n] < px)
				j = 1;
			if (posyd[n] < py)
				j += 2;
			if (poszd[n] < pz)
				j += 4;
			ch = childd[n * 8 + j];
		}

		if (ch != -2) { // skip if child pointer is locked and try again later
			locked = n * 8 + j;
			if (ch == atomicCAS((int *) &childd[locked], ch, -2)) { // try to lock
				if (ch == -1) {
					// if null, just insert the new body
					childd[locked] = i;
				} else { // there already is a body in this position
					patch = -1;
					// create new cell(s) and insert the old and new body
					do {
						depth++;

						cell = atomicSub((int *) &bottomd, 1) - 1;
						if (cell <= nbodiesd) {
							*errd = 1;
							bottomd = nnodesd;
						}
						patch = max(patch, cell);

						x = (j & 1) * r;
						y = ((j >> 1) & 1) * r;
						z = ((j >> 2) & 1) * r;
						r *= 0.5f;

						massd[cell] = -1.0f;
						startd[cell] = -1;
						x = posxd[cell] = posxd[n] - r + x;
						y = posyd[cell] = posyd[n] - r + y;
						z = poszd[cell] = poszd[n] - r + z;
						for (k = 0; k < 8; k++)
							childd[cell * 8 + k] = -1;

						if (patch != cell) {
							childd[n * 8 + j] = cell;
						}

						j = 0;
						if (x < posxd[ch])
							j = 1;
						if (y < posyd[ch])
							j += 2;
						if (z < poszd[ch])
							j += 4;
						childd[cell * 8 + j] = ch;

						n = cell;
						j = 0;
						if (x < px)
							j = 1;
						if (y < py)
							j += 2;
						if (z < pz)
							j += 4;

						ch = childd[n * 8 + j];
						// repeat until the two bodies are different children
					} while (ch >= 0);
					childd[n * 8 + j] = i;
					__threadfence(); // push out subtree
					childd[locked] = patch;
				}

				localmaxdepth = max(depth, localmaxdepth);
				i += inc; // move on to next body
				skip = 1;
			}
		}
		__syncthreads(); // throttle
	}
	// record maximum tree depth
	atomicMax((int *) &maxdepthd, localmaxdepth);
}

/******************************************************************************/
/*** compute center of mass ***************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernelGPU() {
	register int i, j, k, ch, inc, missing, cnt, bottom;
	register float m, cm, px, py, pz;
	__shared__ volatile int child[THREADS3 * 8];

	bottom = bottomd;
	inc = blockDim.x * gridDim.x;
	k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x; // align to warp size
	if (k < bottom)
		k += inc;

	missing = 0;
	// iterate over all cells assigned to thread
	while (k <= nnodesd) {
		if (missing == 0) {
			// new cell, so initialize
			cm = 0.0f;
			px = 0.0f;
			py = 0.0f;
			pz = 0.0f;
			cnt = 0;
			j = 0;
			for (i = 0; i < 8; i++) {
				ch = childd[k * 8 + i];
				if (ch >= 0) {
					if (i != j) {
						// move children to front (needed later for speed)
						childd[k * 8 + i] = -1;
						childd[k * 8 + j] = ch;
					}
					child[missing * THREADS3 + threadIdx.x] = ch; // cache missing children
					m = massd[ch];
					missing++;
					if (m >= 0.0f) {
						// child is ready
						missing--;
						if (ch >= nbodiesd) { // count_iterations bodies (needed later)
							cnt += countd[ch] - 1;
						}
						// add child's contribution
						cm += m;
						px += posxd[ch] * m;
						py += posyd[ch] * m;
						pz += poszd[ch] * m;
					}
					j++;
				}
			}
			cnt += j;
		}

		if (missing != 0) {
			do {
				// poll missing child
				ch = child[(missing - 1) * THREADS3 + threadIdx.x];
				m = massd[ch];
				if (m >= 0.0f) {
					// child is now ready
					missing--;
					if (ch >= nbodiesd) {
						// count_iterations bodies (needed later)
						cnt += countd[ch] - 1;
					}
					// add child's contribution
					cm += m;
					px += posxd[ch] * m;
					py += posyd[ch] * m;
					pz += poszd[ch] * m;
				}
				// repeat until we are done or child is not ready
			} while ((m >= 0.0f) && (missing != 0));
		}

		if (missing == 0) {
			// all children are ready, so store computed information
			countd[k] = cnt;
			m = 1.0f / cm;
			posxd[k] = px * m;
			posyd[k] = py * m;
			poszd[k] = pz * m;
			__threadfence(); // make sure data are visible before setting mass
			massd[k] = cm;
			k += inc; // move on to next cell
		}
	}
}

/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS4, FACTOR4)
void SortKernelGPU() {
	register int i, k, ch, dec, start, bottom;

	bottom = bottomd;
	dec = blockDim.x * gridDim.x;
	k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

	// iterate over all cells assigned to thread
	while (k >= bottom) {
		start = startd[k];
		if (start >= 0) {
			for (i = 0; i < 8; i++) {
				ch = childd[k * 8 + i];
				if (ch >= nbodiesd) {
					// child is a cell
					startd[ch] = start; // set start ID of child
					start += countd[ch]; // add #bodies in subtree
				} else if (ch >= 0) {
					// child is a body
					sortd[start] = ch; // record body in 'sorted' array
					start++;
				}
			}
			k -= dec; // move on to next cell
		}
		__syncthreads(); // throttle
	}
}

/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernelGPU() {
	register int i, j, k, n, depth, base, sbase, diff, t;
	register float px, py, pz, ax, ay, az, dx, dy, dz, tmp;
	__shared__ volatile int pos[MAXDEPTH * THREADS5 / WARPSIZE], node[MAXDEPTH * THREADS5 / WARPSIZE];
	__shared__ float dq[MAXDEPTH * THREADS5 / WARPSIZE];

	if (0 == threadIdx.x) {
		tmp = radiusd;
		// precompute values that depend only on tree level
		dq[0] = tmp * tmp * itolsqd;
		for (i = 1; i < maxdepthd; i++) {
			dq[i] = dq[i - 1] * 0.25f;
			dq[i - 1] += epssqd;
		}
		dq[i - 1] += epssqd;

		if (maxdepthd > MAXDEPTH) {
			*errd = maxdepthd;
		}
	}
	__syncthreads();

	if (maxdepthd <= MAXDEPTH) {
		// figure out first thread in each warp (lane 0)
		base = threadIdx.x / WARPSIZE;
		sbase = base * WARPSIZE;
		j = base * MAXDEPTH;

		diff = threadIdx.x - sbase;
		// make multiple copies to avoid index calculations later
		if (diff < MAXDEPTH) {
			dq[diff + j] = dq[diff];
		}
		__syncthreads();

		// iterate over all bodies assigned to thread
		for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd;
				k += blockDim.x * gridDim.x) {
			i = sortd[k]; // get permuted/sorted index
			// cache position info
			px = posxd[i];
			py = posyd[i];
			pz = poszd[i];

			ax = 0.0f;
			ay = 0.0f;
			az = 0.0f;

			// initialize iteration stack, i.e., push root node onto stack
			depth = j;
			if (sbase == threadIdx.x) {
				node[j] = nnodesd;
				pos[j] = 0;
			}

			while (depth >= j) {
				// stack is not empty
				while ((t = pos[depth]) < 8) {
					// node on top of stack has more children to process
					n = childd[node[depth] * 8 + t]; // load child pointer
					if (sbase == threadIdx.x) {
						// I'm the first thread in the warp
						pos[depth] = t + 1;
					}
					if (n >= 0) {
						dx = posxd[n] - px;
						dy = posyd[n] - py;
						dz = poszd[n] - pz;
						tmp = dx * dx + (dy * dy + (dz * dz + epssqd)); // compute distance squared (plus softening)
						if ((n < nbodiesd) || __all(tmp >= dq[depth])) { // check if all threads agree that cell is far enough away (or is a body)
							tmp = rsqrtf(tmp); // compute distance
							tmp = massd[n] * tmp * tmp * tmp;
							ax += dx * tmp;
							ay += dy * tmp;
							az += dz * tmp;
						//	atomicAdd(count_iterations, 1);
						} else {
							// push cell onto stack
							depth++;
							if (sbase == threadIdx.x) {
								node[depth] = n;
								pos[depth] = 0;
							}
						}
					} else {
						depth = max(j, depth - 1); // early out because all remaining children are also zero
					}
				}
				depth--; // done with this level
			}

			if (stepd > 0) {
				// update velocity
				velxd[i] += (ax - accxd[i]) * dthfd;
				velyd[i] += (ay - accyd[i]) * dthfd;
				velzd[i] += (az - acczd[i]) * dthfd;
			}

			// save computed acceleration
			accxd[i] = ax;
			accyd[i] = ay;
			acczd[i] = az;
		}
	}
}

/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS6, FACTOR6)
void IntegrationKernelGPU() {
	register int i, inc;
	register float dvelx, dvely, dvelz;
	register float velhx, velhy, velhz;

	// iterate over all bodies assigned to thread
	inc = blockDim.x * gridDim.x;
	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc) {
		// integrate
		dvelx = accxd[i] * dthfd;
		dvely = accyd[i] * dthfd;
		dvelz = acczd[i] * dthfd;

		velhx = velxd[i] + dvelx;
		velhy = velyd[i] + dvely;
		velhz = velzd[i] + dvelz;

		posxd[i] += velhx * dtimed;
		posyd[i] += velhy * dtimed;
		poszd[i] += velhz * dtimed;

		velxd[i] = velhx + dvelx;
		velyd[i] = velhy + dvely;
		velzd[i] = velhz + dvelz;
	}
}

/******************************************************************************/

static void CudaTest(char *msg) {
	cudaError_t e;

	cudaThreadSynchronize();
	if (cudaSuccess != (e = cudaGetLastError())) {
		fprintf(stderr, "%s: %d\n", msg, e);
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		exit(-1);
	}
}

/******************************************************************************/

// random number generator
#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;

static void drndset(int seed) {
	A = 1;
	B = 0;
	randx = (A * seed + B) & MASK;
	A = (MULT * A) & MASK;
	B = (MULT * B + ADD) & MASK;
}

static double drnd() {
	lastrand = randx;
	randx = (A * randx + B) & MASK;
	return (double) lastrand / TWOTO31;
}

static void genInput(unsigned int nbodies, float* mass, float* posx,
		float* posy, float *posz, float* velx, float* vely, float *velz) {

	register double rsc, vsc, r, v, x, y, z, sq, scale;
	drndset(7);
	rsc = (3 * 3.1415926535897932384626433832795) / 16;
	vsc = sqrt(1.0 / rsc);
	for (int i = 0; i < nbodies; i++) {
		mass[i] = 1.0 / nbodies;
		r = 1.0 / sqrt(pow(drnd() * 0.999, -2.0 / 3.0) - 1);
		do {
			x = drnd() * 2.0 - 1.0;
			y = drnd() * 2.0 - 1.0;
			z = drnd() * 2.0 - 1.0;
			sq = x * x + y * y + z * z;
		} while (sq > 1.0);

        scale = rsc * r / sqrt(sq);
		posx[i] = x * scale;
		posy[i] = y * scale;
		posz[i] = z * scale;

		do {
			x = drnd();
			y = drnd() * 0.1;
		} while (y > x * x * pow(1 - x * x, 3.5));
		v = x * sqrt(2.0 / sqrt(1 + r * r));
		do {
			x = drnd() * 2.0 - 1.0;
			y = drnd() * 2.0 - 1.0;
			z = drnd() * 2.0 - 1.0;
			sq = x * x + y * y + z * z;
		} while (sq > 1.0);
		scale = vsc * v / sqrt(sq);
		velx[i] = x * scale; // x * scale;
		vely[i] = y * scale; // y * scale;
		velz[i] = z * scale; // z * scale;
	}
}

void loadData(char* filename, unsigned int nbodies, float* mass, float* posx,
		float* posy, float *posz, float* velx, float* vely, float *velz)
{

    float 	scaleFactor = 1.5f/10000.0f;		// 10.0f, 50
    float 	velFactor = 8.0f/100000.0f;			// 15.0f, 100
    float	massFactor = 12.0f;	// 50000000.0,

    int skip = 49152 / nbodies;
    FILE *fin;

    if ((fin = fopen(filename, "r")))
    {
        char buf[256];
        float v[7];
        int idx = 0;

        // allocate memory

        int k=0;
        for (int i=0; i< nbodies; i++,k++)
        {
            // depend on input size... skip lines
            for (int j=0; j < skip; j++,k++)
                fgets (buf, 256, fin);	// lead line

            sscanf(buf, "%f %f %f %f %f %f %f", v+0, v+1, v+2, v+3, v+4, v+5, v+6);

            // update index
            idx = i;

            // position
            posx[idx] = v[1]*scaleFactor;
            posy[idx] = v[2]*scaleFactor;
            posx[idx] = v[3]*scaleFactor;

            // mass
            mass[idx] = v[0]*massFactor;

            // velocity
            velx[idx] = v[4]*velFactor;
            vely[idx] = v[5]*velFactor;
            velz[idx] = v[6]*velFactor;
        }
    }
    else
    {
        printf("cannot find file...: %s\n", filename);
        exit(0);
    }
}

/******************************************************************************/

Point3D* particles;

void moveToVBO(float* posx, float* posy, float* posz){
#pragma parallel omp for
    for (unsigned long i=0; i<NBODIES; i++){
        particles[i].x=posx[i];
        particles[i].y=posy[i];
        particles[i].z=posz[i];
    }
}


void moveToVBO2(float* posx, float* posy, float* posz){
#pragma parallel omp for
    for (unsigned long i=0; i<NBODIES; i++){
        particles[i].x=posx[i];
        particles[i].y=posy[i];
        particles[i].z=posz[i];
    }
}

void main_loop() {
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    starttime = clock();
    cudaEventRecord(start, 0);
    InitializationKernelGPU<<<1, 1>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    timing[0] += time;
    CudaTest("kernel 0 launch failed");


    cudaEventRecord(start, 0);
    BoundingBoxKernelGPU<<<blocks * FACTOR1, THREADS1>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    timing[1] += time;
    CudaTest("kernel 1 launch failed");

    cudaEventRecord(start, 0);
    TreeBuildingKernelGPU<<<blocks * FACTOR2, THREADS2>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    timing[2] += time;
    CudaTest("kernel 2 launch failed");

    cudaEventRecord(start, 0);
    SummarizationKernelGPU<<<blocks * FACTOR3, THREADS3>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    timing[3] += time;
    CudaTest("kernel 3 launch failed");

    cudaEventRecord(start, 0);
    SortKernelGPU<<<blocks * FACTOR4, THREADS4>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    timing[4] += time;
    CudaTest("kernel 4 launch failed");
	float *gradius;
	float radius;
	cudaGetSymbolAddress((void**) &gradius, radiusd);
	cudaMemcpy(&radius, gradius, sizeof(float), cudaMemcpyDeviceToHost);
	int *deth;
	int maxd;
	cudaGetSymbolAddress((void**) &deth, maxdepthd);
	cudaMemcpy(&maxd, deth, sizeof(int), cudaMemcpyDeviceToHost);
	int *stepp;
	cudaGetSymbolAddress((void**) &stepp, stepd);
	cudaMemcpy(&step, stepp, sizeof(int), cudaMemcpyDeviceToHost);
    CudaTest("Get 4 launch failed");

    //printf("R: %f %d\n",radius,maxd);
    cudaDeviceSynchronize();
#ifdef MB_REF
			double f_start = getTimeMS();
			ForceCalculationKernelGPU<<<blocks * FACTOR5, THREADS5>>>();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			timing[5] += getTimeMS() - f_start;
			CudaTest("kernel 5 launch failed");
#else
			//register int ii = 0;
			double fk_start = getTimeMS();
			BHForce* fc = createForce(
                    m, 
                    px, py, pz, 
                    vx, vy, vz, 
                    ax, ay, az, 
                    ch, st, 
                    0, (unsigned long) nbodies, (unsigned long) step,
					dq, nnodes, maxd , radius, citl);

			//rs->synchronize();
			rs->submit(fc);
			rs->synchronize();
			double f_start = getTimeMS();
			timing[5] += getTimeMS() - fk_start;
			deleteForce(fc);
			cudaGetLastError();
			CudaTest("kernel 5 launch failed");
#endif


    cudaEventRecord(start, 0);
    IntegrationKernelGPU<<<blocks * FACTOR6, THREADS6>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    timing[6] += time;
    CudaTest("kernel 6 launch failed");

#ifdef DISPLAY
	moveToVBO2(px,py,pz);
	unsigned int size = NBODIES * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, (const GLvoid*) particles, GL_DYNAMIC_DRAW);
	aux_Draw();
#endif
    step++;
}


int burtcher(int argc, char* argv[], RuntimeScheduler *rsh, unsigned int bodies, unsigned int timesteps, float *massl, float *posxl, float *posyl, float *poszl, int *childl) {
    register int i, run;
    register float dtime, dthf, epssq, itolsq;

    nbodies = bodies;
    rs = rsh;

    int *errl, *sortl, *countl, *startl;

    float *velxl, *velyl, *velzl;
    float *accxl, *accyl, *acczl;
    float *maxxl, *maxyl, *maxzl;
    float *minxl, *minyl, *minzl;
    //	register double rsc, vsc, r, v, x, y, z, sq, scale;

    // Display

    particles = new Point3D[nbodies];

#ifdef DISPLAY
	if (0 == initGL(&argc, argv)) {
		return 0;
	}
#endif
#ifdef DISPLAY
    createVBO(&vbo, (const GLvoid*) particles);
    glutDisplayFunc(main_loop);
#endif



    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
        fprintf(stderr, "There is no CUDA capable device\n");
        exit(-1);
    }
    printf("Using gpu : %s \n", deviceProp.name);
    if (deviceProp.major < 2) {
        fprintf(stderr, "Need at least compute capability 2.0\n");
        exit(-1);
    }
    if (deviceProp.warpSize != WARPSIZE) {
        fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
        exit(-1);
    }

    blocks = deviceProp.multiProcessorCount;
    fprintf(stderr, "blocks = %d\n", blocks);

    if ((WARPSIZE <= 0) || (WARPSIZE & (WARPSIZE - 1) != 0)) {
        fprintf(stderr,
                "Warp size must be greater than zero and a power of two\n");
        exit(-1);
    }
    if (MAXDEPTH > WARPSIZE) {
        fprintf(stderr, "MAXDEPTH must be less than or equal to WARPSIZE\n");
        exit(-1);
    }
    if ((THREADS1 <= 0) || (THREADS1 & (THREADS1 - 1) != 0)) {
        fprintf(stderr,
                "THREADS1 must be greater than zero and a power of two\n");
        exit(-1);
    }

    // set L1/shared memory configuration
    cudaFuncSetCacheConfig(BoundingBoxKernelGPU, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(TreeBuildingKernelGPU, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(SummarizationKernelGPU, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(SortKernelGPU, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ForceCalculationKernelGPU, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(IntegrationKernelGPU, cudaFuncCachePreferL1);

    cudaHostAlloc((void**) &dq, MAXDEPTH * sizeof(float), cudaHostAllocPortable);

    cudaGetLastError(); // reset error value
    run = 0;
    //for (run = 0; run < 3; run++) {
    for (i = 0; i < 7; i++)
        timing[i] = 0.0f;

    nnodes = nbodies * 2;
    if (nnodes < 1024 * blocks)
        nnodes = 1024 * blocks;
    while ((nnodes & (WARPSIZE - 1)) != 0)
        nnodes++;
    nnodes--;

    //    timesteps = atoi(argv[2]);
    dtime = 0.025;
    dthf = dtime * 0.5f;
    epssq = 0.05 * 0.05;
    itolsq = 1.0f / (0.5 * 0.5);

    // allocate memory
#ifndef DISPLAY
    for(run=0; run < 3; run++) {
#endif
    if (run == 0) {
        fprintf(stderr, "nodes = %d\n", nnodes + 1);
        fprintf(stderr, "configuration: %d bodies, %d time steps\n",
                nbodies, timesteps);

        if (cudaSuccess != cudaMalloc((void **) &citl, sizeof(int)))
            fprintf(stderr, "could not allocate citl\n");
        CudaTest("couldn't allocate citl");
        if (cudaSuccess != cudaMalloc((void **) &errl, sizeof(int)))
            fprintf(stderr, "could not allocate errd\n");
        CudaTest("couldn't allocate errd");
        if (cudaSuccess
                != cudaMalloc((void **) &countl,
                    sizeof(int) * (nnodes + 1)))
            fprintf(stderr, "could not allocate countd\n");
        CudaTest("couldn'tnbodiesd allocate countd");
        if (cudaSuccess
                != cudaMalloc((void **) &startl,
                    sizeof(int) * (nnodes + 1)))
            fprintf(stderr, "could not allocate startd\n");
        CudaTest("couldn't allocate startd");

        // alias arrays
        int inc = ((int) nbodies + WARPSIZE - 1) & (-WARPSIZE);
        velxl = (float *) &childl[0 * inc];
        velyl = (float *) &childl[1 * inc];
        velzl = (float *) &childl[2 * inc];
        accxl = (float *) &childl[3 * inc];
        accyl = (float *) &childl[4 * inc];
        acczl = (float *) &childl[5 * inc];
        sortl = (int *) &childl[6 * inc];

        if (cudaSuccess
                != cudaMalloc((void **) &maxxl, sizeof(float) * blocks))
            fprintf(stderr, "could not allocate maxxd\n");
        CudaTest("couldn't allocate maxxd");
        if (cudaSuccess
                != cudaMalloc((void **) &maxyl, sizeof(float) * blocks))
            fprintf(stderr, "could not allocate maxyd\n");
        CudaTest("couldn't allocate maxyd");
        if (cudaSuccess
                != cudaMalloc((void **) &maxzl, sizeof(float) * blocks))
            fprintf(stderr, "could not allocate maxzd\n");
        CudaTest("couldn't allocate maxzd");
        if (cudaSuccess
                != cudaMalloc((void **) &minxl, sizeof(float) * blocks))
            fprintf(stderr, "could not allocate minxd\n");
        CudaTest("couldn't allocate minxd");
        if (cudaSuccess
                != cudaMalloc((void **) &minyl, sizeof(float) * blocks))
            fprintf(stderr, "could not allocate minyd\n");
        CudaTest("couldn't allocate minyd");
        if (cudaSuccess
                != cudaMalloc((void **) &minzl, sizeof(float) * blocks))
            fprintf(stderr, "could not allocate minzd\n");
        CudaTest("couldn't allocate minzd");

        if (cudaSuccess
                != cudaMemcpyToSymbol(nnodesd, &nnodes, sizeof(int)))
            fprintf(stderr, "copying of nnodes to device failed\n");
        CudaTest("nnode copy to device failed");
        if (cudaSuccess
                != cudaMemcpyToSymbol(nbodiesd, &nbodies, sizeof(int)))
            fprintf(stderr, "copying of nbodies to device failed\n");
        CudaTest("nbody copy to device failed");
        if (cudaSuccess
                != cudaMemcpyToSymbol(count_iterations, &citl, sizeof(void*)))
            fprintf(stderr, "copying of err to device failed\n");
        CudaTest("err copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(errd, &errl, sizeof(void*)))
            fprintf(stderr, "copying of err to device failed\n");
        CudaTest("err copy to device failed");
        if (cudaSuccess
                != cudaMemcpyToSymbol(dtimed, &dtime, sizeof(float)))
            fprintf(stderr, "copying of dtime to device failed\n");
        CudaTest("dtime copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(dthfd, &dthf, sizeof(float)))
            fprintf(stderr, "copying of dthf to device failed\n");
        CudaTest("dthf copy to device failed");
        if (cudaSuccess
                != cudaMemcpyToSymbol(epssqd, &epssq, sizeof(float)))
            fprintf(stderr, "copying of epssq to device failed\n");
        CudaTest("epssq copy to device failed");
        if (cudaSuccess
                != cudaMemcpyToSymbol(itolsqd, &itolsq, sizeof(float)))
            fprintf(stderr, "copying of itolsq to device failed\n");
        CudaTest("itolsq copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(sortd, &sortl, sizeof(void*)))
            fprintf(stderr, "copying of sortl to device failed\n");
        CudaTest("sortl copy to device failed");
        if (cudaSuccess
                != cudaMemcpyToSymbol(countd, &countl, sizeof(void*)))
            fprintf(stderr, "copying of countl to device failed\n");
        CudaTest("countl copy to device failed");
        if (cudaSuccess
                != cudaMemcpyToSymbol(startd, &startl, sizeof(void*)))
            fprintf(stderr, "copying of startl to device failed\n");
        CudaTest("startl copy to device failed");
        if (cudaSuccess
                != cudaMemcpyToSymbol(childd, &childl, sizeof(void*)))
            fprintf(stderr, "copying of childl to device failed\n");
        CudaTest("childl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(massd, &massl, sizeof(void*)))
            fprintf(stderr, "copying of massl to device failed\n");
        CudaTest("massl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(posxd, &posxl, sizeof(void*)))
            fprintf(stderr, "copying of posxl to device failed\n");
        CudaTest("posxl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(posyd, &posyl, sizeof(void*)))
            fprintf(stderr, "copying of posyl to device failed\n");
        CudaTest("posyl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(poszd, &poszl, sizeof(void*)))
            fprintf(stderr, "copying of poszl to device failed\n");
        CudaTest("poszl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(velxd, &velxl, sizeof(void*)))
            fprintf(stderr, "copying of velxl to device failed\n");
        CudaTest("velxl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(velyd, &velyl, sizeof(void*)))
            fprintf(stderr, "copying of velyl to device failed\n");
        CudaTest("velyl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(velzd, &velzl, sizeof(void*)))
            fprintf(stderr, "copying of velzl to device failed\n");
        CudaTest("velzl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(accxd, &accxl, sizeof(void*)))
            fprintf(stderr, "copying of accxl to device failed\n");
        CudaTest("accxl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(accyd, &accyl, sizeof(void*)))
            fprintf(stderr, "copying of accyl to device failed\n");
        CudaTest("accyl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(acczd, &acczl, sizeof(void*)))
            fprintf(stderr, "copying of acczl to device failed\n");
        CudaTest("acczl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(maxxd, &maxxl, sizeof(void*)))
            fprintf(stderr, "copying of maxxl to device failed\n");
        CudaTest("maxxl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(maxyd, &maxyl, sizeof(void*)))
            fprintf(stderr, "copying of maxyl to device failed\n");
        CudaTest("maxyl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(maxzd, &maxzl, sizeof(void*)))
            fprintf(stderr, "copying of maxzl to device failed\n");
        CudaTest("maxzl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(minxd, &minxl, sizeof(void*)))
            fprintf(stderr, "copying of minxl to device failed\n");
        CudaTest("minxl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(minyd, &minyl, sizeof(void*)))
            fprintf(stderr, "copying of minyl to device failed\n");
        CudaTest("minyl copy to device failed");
        if (cudaSuccess != cudaMemcpyToSymbol(minzd, &minzl, sizeof(void*)))
            fprintf(stderr, "copying of minzl to device failed\n");
        CudaTest("minzl copy to device failed");
    }

    genInput(nbodies, massl, posxl, posyl, poszl, velxl, velyl, velzl);

    //loadData("bin/dubinski.tab",nbodies, massl, posxl, posyl, poszl, velxl, velyl, velzl);

	m=massl;
	px=posxl; py=posyl; pz=poszl;
	vx=velxl; vy=velyl; vz=velzl;
	ax=accxl; ay=accyl; az=acczl;
	ch=childl;
	st=sortl;

#ifdef DISPLAY
	glutMainLoop();
#else
	for(int ii=0; ii < timesteps; ii++) main_loop();

    cudaMemset(citl, 0, sizeof(int));
    CudaTest("kernel launch failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int error;
    // transfer result back to CPU
    if (cudaSuccess
            != cudaMemcpy(&error, errl, sizeof(int),
                cudaMemcpyDeviceToHost))
        fprintf(stderr, "copying of err from device failed\n");
    CudaTest("err copy from device failed");

    fprintf(stderr, "runtime: (");
    float time = 0;
    for (i = 1; i < 7; i++) {
        fprintf(stderr, " %.1f ", timing[i] / (float)timesteps);
        time += timing[i] / (float)timesteps;
        timing[i]=0.f;
    }
    if (error == 0) {
        fprintf(stderr, ") = %.1f\n", time);
    } else {
        fprintf(stderr, ") = %.1f FAILED %d\n", time, error);
    }
}
#endif

#ifdef MB_COUNT_IT
    int c = 0;
    cudaMemcpy(&c, citl, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(citl, 0, sizeof(int));
    printf("It: %d\n", c);
#endif


    cudaMemset(citl, 0, sizeof(int));

#ifdef MB_COUNT_IT
    int c = 0;
    cudaMemcpy(&c, citl, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(citl, 0, sizeof(int));
    printf("It: %d\n", c);
#endif
    CudaTest("kernel launch failed");
    printf("%.2e %.2e %.2e\n", posxl[i], posyl[i], poszl[i]);

    cudaFree(errl);
    cudaFree(countl);
    cudaFree(startl);

    cudaFree(maxxl);
    cudaFree(maxyl);
    cudaFree(maxzl);
    cudaFree(minxl);
    cudaFree(minyl);
    cudaFree(minzl);


    delete rs;
    return 0;
}
#endif
