/*
 * BHForce.x86.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: jbarbosa
 */
#include <config/common.h>
#include <gamalib/gamalib.h>
#include <gamalib/gamalib.cuh>

#if (SAMPLE == 4 || SAMPLE==5)

#include "BHForce.h"

template<>
void __DEVICE__ BHForce::execute<CPU_X86>() {
    register int i, j, k, n, depth, base, sbase, diff, t;
    register float px, py, pz, ax, ay, az, dx, dy, dz, tmp;
    register unsigned long rupper = upper;
    register unsigned long limit = TID_SIZE;

    volatile int pos[MAXDEPTH * 8], node[MAXDEPTH * 8];
    float dq[MAXDEPTH * 8];

    smartPtr<float> posxl = posx; smartPtr<float> posyl = posy; smartPtr<float> poszl = posz;
    smartPtr<float> velxl = vely; smartPtr<float> velyl = vely; smartPtr<float> velzl = vely;
    smartPtr<float> accxl = accx; smartPtr<float> accyl = accy; smartPtr<float> acczl = accz;

    smartPtr<float> massl = mass;
    smartPtr<int> childl = child;
    smartPtr<int> sortl = sort;

    register unsigned long nnodesl = nnodes;
    tmp = radius;
    // precompute values that depend only on tree level
    dq[0] = tmp * tmp * 1.0f / (0.5 * 0.5);
    for (i = 1; i < maxdepth; i++) {
        dq[i] = dq[i - 1] * 0.25f;
        dq[i - 1] += epssq;
    }
    dq[i - 1] += epssq;


    if (maxdepth <= MAXDEPTH) {
        // figure out first thread in each warp (lane 0)
        base = TID;
//        sbase = base * 32;
        j = 0;

//        diff = 0;
        // make multiple copies to avoid index calculations later
//        if (diff < MAXDEPTH) {
//            dq[diff + j] = dq[diff];
//        }

        // iterate over all bodies assigned to thread

        for (k = TID + lower; k < rupper; k += limit) {
            i = sortl[k]; // get permuted/sorted index
            // cache position info
            px = posxl[i];
            py = posyl[i];
            pz = poszl[i];

            ax = 0.0f;
            ay = 0.0f;
            az = 0.0f;

            // initialize iteration stack, i.e., push root node onto stack
            depth = j;
            if (sbase == TID) {
                node[j] = nnodesl;
                pos[j] = 0;
            }

            while (depth >= j) {
                // stack is not empty
                while ((t = pos[depth]) < 8) {
                    // node on top of stack has more children to process
                    n = childl[node[depth] * 8 + t]; // load child pointer
                    //if (sbase == TID) {
                        // I'm the first thread in the warp
                        pos[depth] = t + 1;
                    //}
                    if (n >= 0) {
                        dx = posxl[n] - px;
                        dy = posyl[n] - py;
                        dz = poszl[n] - pz;
                        tmp = dx * dx + (dy * dy + (dz * dz + epssq)); // compute distance squared (plus softening)
                        if ((n < NBODIES) || __all(tmp >= dq[depth])) { // check if all threads agree that cell is far enough away (or is a body)
                            tmp = 1.f / sqrt(tmp); // compute distance
                            tmp = massl[n] * tmp * tmp * tmp;
                            ax += dx * tmp;
                            ay += dy * tmp;
                            az += dz * tmp;
                            //							    atomicAdd(count_iterations, 1);
                        } else {
                            // push cell onto stack
                            depth++;
                            //if (sbase == TID) {
                                node[depth] = n;
                                pos[depth] = 0;
                            //}
                        }
                    } else {
                        depth = max(j, depth - 1); // early out because all remaining children are also zero
                    }
                }
                depth--; // done with this level
            }

            if (step > 0) {
                // update velocity
                velxl[i] += (ax - accxl[i]) * dthf;
                velyl[i] += (ay - accyl[i]) * dthf;
                velzl[i] += (az - acczl[i]) * dthf;
            }

            // save computed acceleration
            accxl[i] = ax;
            accyl[i] = ay;
            acczl[i] = az;
        }
    }
}
#endif
