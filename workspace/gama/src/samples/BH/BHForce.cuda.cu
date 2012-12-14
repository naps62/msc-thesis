/*
 * BHForce.cu
 *
 *  Created on: Sep 16, 2012
 *      Author: jbarbosa
 */

#include <config/common.h>
#include <gamalib/gamalib.h>
#include <gamalib/gamalib.cuh>

#if (SAMPLE==4 || SAMPLE==5)

#include "BHForce.h"

//extern __constant__ int nnodesd, nbodiesd;
//extern __constant__ float dtimed, itolsqd;
//, dthfd, epssqd, itolsqd;
//extern __constant__ volatile float *massd;
//, *posxd, *posyd, *poszd, *velxd, *velyd, *velzd, *accxd, *accyd, *acczd;
//extern __constant__ volatile float *maxxd, *maxyd, *maxzd, *minxd, *minyd, *minzd;
//extern __constant__ volatile int *errd, *sortd, *childd, *countd, *startd;
//extern __constant__ int *count_iterations;
//extern __device__ volatile int stepd, bottomd, maxdepthd, blkcntd;
//extern __device__ volatile float radiusd;

#define THREADS5 256
#define WARPSIZE 32

template<>
void __DEVICE__ BHForce::execute<GPU_CUDA>() {
	smartPtr<float> posxd = posx;
	smartPtr<float> posyd = posy;
	smartPtr<float> poszd = posz;
	smartPtr<float> velxd = vely;
	smartPtr<float> velyd = vely;
	smartPtr<float> velzd = velz;
	smartPtr<float> accxd = accx;
	smartPtr<float> accyd = accy;
	smartPtr<float> acczd = accz;

	smartPtr<float> massd = mass;
	smartPtr<int> childd = child;
	smartPtr<int> sortd = sort;
	int nnodesd = nnodes, nbodiesd = NBODIES;
	float radiusd = radius;
	int maxdepthd = maxdepth;
	register int i, j, k, n, depth, base, sbase, diff, t;
	register float px, py, pz, ax, ay, az, dx, dy, dz, tmp;
	__shared__ volatile int pos[MAXDEPTH * THREADS5 / WARPSIZE], node[MAXDEPTH
			* THREADS5 / WARPSIZE];
	__shared__ float dq[MAXDEPTH * THREADS5 / WARPSIZE];

	if (0 == threadIdx.x) {
		tmp = radiusd;
		// precompute values that depend only on tree level
		dq[0] = tmp * tmp * (1.0f / (0.5f * 0.5f));
		for (i = 1; i < maxdepthd; i++) {
			dq[i] = dq[i - 1] * 0.25f;
			dq[i - 1] += epssq;
		}
		dq[i - 1] += epssq;

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
		for (k = TID + lower; k < upper; k += TID_SIZE) {
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
						tmp = dx * dx + (dy * dy + (dz * dz + epssq)); // compute distance squared (plus softening)
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

			if (step > 0) {
				// update velocity
				velxd[i] += (ax - accxd[i]) * dthf;
				velyd[i] += (ay - accyd[i]) * dthf;
				velzd[i] += (az - acczd[i]) * dthf;
			}

			// save computed acceleration
			accxd[i] = ax;
			accyd[i] = ay;
			acczd[i] = az;
		}
	}

}
#endif
