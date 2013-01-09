/*
 * problem.cfg.h
 *
 *  Created on: Sep 16, 2012
 *      Author: jbarbosa
 */

#ifndef PROBLEM_CFG_H_
#define PROBLEM_CFG_H_

#define MPI 3.1415926535897931

#ifndef NSIZE
#define NSIZE 15
#endif

const unsigned long N = 1l << NSIZE;
const unsigned long NN = 64;


const unsigned long ARRAY_SIZE = 1l << 19;
const unsigned long SPLIT = 1024*16;

const unsigned long NKERNELS = 2;

#ifndef NBODY
#define NBODY 15
#endif

#ifndef NTIMESTEPS
#define NTIMESTEPS 100
#endif

const unsigned long NBODIES =  1l << NBODY;
const unsigned long NNODES = (NBODIES * 2) - 1;
const unsigned int TIMESTEPS = NTIMESTEPS;
const float dtime = 0.025f;
const float dthf = dtime * 0.5f;
const float epssq = 0.05 * 0.05;

#define MAXDEPTH 32


#endif /* PROBLEM_CFG_H_ */
